import os
import asyncio
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
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

executor = ThreadPoolExecutor(max_workers=2)

class OpenCLIPEmbeddings(Embeddings):
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def embed_documents(self, texts):
        text_tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().tolist()[0]


def encode_image_to_base64(image_path, max_size=(1024, 1024)):
    """Resizes and compresses image to optimize payload size."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def sync_init_and_process(pdf_path, pdf_name):
    embedding_function = OpenCLIPEmbeddings()
    images_dir = f"./pdf_screenshots_{hash(pdf_name)}"
    os.makedirs(images_dir, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=200)
    db_path = f"./chroma_streamlit_{hash(pdf_name)}"
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function
    )

    ids, embeddings_list, metadatas, texts = [], [], [], []
    for idx, image in enumerate(images):
        page_num = idx + 1
        img_path = os.path.join(images_dir, f"page_{page_num}.jpg")
        image.save(img_path, "JPEG")
        img_emb = embedding_function.embed_image(img_path)
        
        ids.append(f"page_{page_num}")
        embeddings_list.append(img_emb)
        metadatas.append({"page": page_num, "image_path": img_path})
        texts.append(f"Page {page_num} screenshot from document {pdf_name}")

    vectorstore._collection.add(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=texts
    )
    return vectorstore, len(images)


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
            st.success(f"Successfully processed & indexed {total_pages} pages!")

    # Query Handling
    query = st.text_input("Ask a question about the PDF:")
    if query and "vectorstore" in st.session_state:
        with st.spinner("Retrieving relevant visual pages & generating answer..."):
            vectorstore = st.session_state.vectorstore
            
            # Step 1: Similarity Search
            relevant_docs = vectorstore.similarity_search(query, k=2)

            # Step 2: Fallback logic for generic or broad queries
            if not relevant_docs:
                all_records = vectorstore._collection.get()
                if all_records and "metadatas" in all_records and len(all_records["metadatas"]) > 0:
                    class FallbackDoc:
                        def __init__(self, metadata):
                            self.metadata = metadata
                    relevant_docs = [FallbackDoc(meta) for meta in all_records["metadatas"][:2]]

            # Step 3: Process Pages and Query Gemini Vision Models with Fallback
            if not relevant_docs:
                st.warning("No relevant pages found in the document for your query.")
            else:
                content_payload = [
                    {
                        "type": "text", 
                        "text": f"Answer the user query strictly using the attached document page images.\n\nUser Query: {query}"
                    }
                ]

                st.write("#### Retrieved Pages:")
                cols = st.columns(len(relevant_docs))
                for idx, doc in enumerate(relevant_docs):
                    img_path = doc.metadata.get("image_path")
                    page_num = doc.metadata.get("page")
                    if img_path and os.path.exists(img_path):
                        base64_img = encode_image_to_base64(img_path)
                        content_payload.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        })
                        with cols[idx]:
                            st.image(img_path, caption=f"Page {page_num}")

                # Candidate Vision models to attempt in sequence
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
                        break  # Stop execution upon first successful model call
                    except Exception as e:
                        last_error = str(e)
                        continue

                if response_content:
                    st.divider()
                    st.subheader("💡 Answer")

                    # Parse output format (Extract text if returned as structured list/dict)
                    parsed_text = ""
                    if isinstance(response_content, list):
                        for block in response_content:
                            if isinstance(block, dict) and "text" in block:
                                parsed_text += block["text"]
                            elif isinstance(block, str):
                                parsed_text += block
                    else:
                        parsed_text = str(response_content)

                    # Replace literal escaped newlines with actual line breaks
                    parsed_text = parsed_text.replace("\\n", "\n")

                    # Render answer inside a clean Streamlit UI card
                    with st.container(border=True):
                        st.markdown(parsed_text)

                    st.caption(f"⚡ *Generated using `{successful_model}`*")
                else:
                    st.error(f"All candidate Vision models failed. Last error: {last_error}")

elif uploaded_file and not gemini_api_key:
    st.info("Please enter your Gemini API Key in the sidebar or Streamlit Secrets to proceed.")
