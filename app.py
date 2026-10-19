import os
import gc
import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path

import torch
import open_clip
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# Page Configuration
st.set_page_config(page_title="Multimodal Visual RAG", layout="wide")
st.title("📚 Multimodal Visual RAG Agent")


@st.cache_resource(show_spinner=False)
def load_openclip_model():
    """Cache OpenCLIP model in memory so it doesn't reload on every rerun."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer, device


class OpenCLIPEmbeddings(Embeddings):
    def __init__(self):
        self.model, self.preprocess, self.tokenizer, self.device = load_openclip_model()

    def embed_documents(self, texts):
        text_tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_image(self, image_path):
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().tolist()[0]


def encode_image_to_base64(image_path, max_size=(800, 800)):
    """Downsamples image size to maintain extremely lightweight payloads."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=75)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def sync_init_and_process(pdf_path, pdf_name):
    embedding_function = OpenCLIPEmbeddings()
    images_dir = f"./pdf_screenshots_{abs(hash(pdf_name))}"
    os.makedirs(images_dir, exist_ok=True)
    
    # Reduced DPI from 200 to 100 to avoid RAM spike on Streamlit Cloud
    images = convert_from_path(pdf_path, dpi=100)
    db_path = f"./chroma_streamlit_{abs(hash(pdf_name))}"
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function
    )

    ids, embeddings_list, metadatas, texts = [], [], [], []
    
    for idx, image in enumerate(images):
        page_num = idx + 1
        img_path = os.path.join(images_dir, f"page_{page_num}.jpg")
        image.save(img_path, "JPEG", quality=80)
        
        # Calculate image embedding
        img_emb = embedding_function.embed_image(img_path)
        
        ids.append(f"page_{page_num}")
        embeddings_list.append(img_emb)
        metadatas.append({"page": page_num, "image_path": img_path})
        texts.append(f"Page {page_num} screenshot from document {pdf_name}")

    # Add to vector database
    vectorstore._collection.add(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=texts
    )
    
    # Cleanup memory
    del images
    gc.collect()
    
    return vectorstore, len(ids)


# Sidebar Credentials Setup
st.sidebar.header("Configuration")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not gemini_api_key:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Main Interface
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file and gemini_api_key:
    # Reset vectorstore state if a new file is uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]

    # Indexing PDF
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF screenshots & indexing visual embeddings..."):
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            vectorstore, total_pages = sync_init_and_process(temp_path, uploaded_file.name)
            st.session_state.vectorstore = vectorstore
            
            # Clean temp pdf file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            st.success(f"Successfully processed & indexed {total_pages} pages!")

    # Direct Q&A Interface
    query = st.text_input("Ask a question about the PDF:")
    if query and "vectorstore" in st.session_state:
        with st.spinner("Analyzing document & generating answer..."):
            vectorstore = st.session_state.vectorstore
            
            # Step 1: Similarity Search
            relevant_docs = vectorstore.similarity_search(query, k=2)

            # Step 2: Fallback logic for generic queries
            if not relevant_docs:
                all_records = vectorstore._collection.get()
                if all_records and "metadatas" in all_records and len(all_records["metadatas"]) > 0:
                    class FallbackDoc:
                        def __init__(self, metadata):
                            self.metadata = metadata
                    relevant_docs = [FallbackDoc(meta) for meta in all_records["metadatas"][:2]]

            # Step 3: Process Pages behind the scenes & Query Gemini
            if not relevant_docs:
                st.warning("No relevant pages found in the document for your query.")
            else:
                content_payload = [
                    {
                        "type": "text", 
                        "text": f"Answer the user query strictly using the attached document page images.\n\nUser Query: {query}"
                    }
                ]

                # Prepare image payload in background
                for doc in relevant_docs:
                    img_path = doc.metadata.get("image_path")
                    if img_path and os.path.exists(img_path):
                        base64_img = encode_image_to_base64(img_path)
                        content_payload.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        })

                # Candidate Vision models sequence
                candidate_models = [
                    "gemini-3.8-flash",
                    "gemini-2.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-2.0-flash-exp"
                ]

                response_content = None
                successful_model = None
                last_error = None

                for model_name in candidate_models:
                    try:
                        llm_vision = ChatGoogleGenerativeAI(
                            google_api_key=gemini_api_key,
                            model=model_name,
                            temperature=0.1
                        )
                        res = llm_vision.invoke([HumanMessage(content=content_payload)])
                        response_content = res.content
                        successful_model = model_name
                        break
                    except Exception as e:
                        last_error = str(e)
                        continue

                if response_content:
                    st.divider()
                    st.subheader("💡 Answer")

                    # Parse output format cleanly
                    parsed_text = ""
                    if isinstance(response_content, list):
                        for block in response_content:
                            if isinstance(block, dict) and "text" in block:
                                parsed_text += block["text"]
                            elif isinstance(block, str):
                                parsed_text += block
                    else:
                        parsed_text = str(response_content)

                    parsed_text = parsed_text.replace("\\n", "\n")

                    # Render answer inside a clean Streamlit UI card
                    with st.container(border=True):
                        st.markdown(parsed_text)

                    st.caption(f"⚡ *Generated using `{successful_model}`*")
                else:
                    st.error(f"All candidate Vision models failed. Last error: {last_error}")

elif uploaded_file and not gemini_api_key:
    st.info("Please enter your Gemini API Key in the sidebar or Streamlit Secrets to proceed.")
