import os
import gc
import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path

import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma

# Page Configuration
st.set_page_config(page_title="Multimodal Visual RAG", layout="wide")
st.title("📚 Multimodal Visual RAG Agent")


def encode_image_to_base64(image_path, max_size=(800, 800)):
    """Downsamples image size to maintain extremely lightweight payloads."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=75)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def sync_init_and_process(pdf_path, pdf_name, openrouter_key):
    # OpenRouter API Embeddings via langchain_community.embeddings.OpenAIEmbeddings
    embedding_function = OpenAIEmbeddings(
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model="nvidia/nemotron-3.5-embed-1b:free"
    )

    images_dir = f"./pdf_screenshots_{abs(hash(pdf_name))}"
    os.makedirs(images_dir, exist_ok=True)
    
    # Render PDF pages to screenshots
    images = convert_from_path(pdf_path, dpi=100)
    db_path = f"./chroma_streamlit_{abs(hash(pdf_name))}"
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_function
    )

    ids, metadatas, texts = [], [], []
    
    for idx, image in enumerate(images):
        page_num = idx + 1
        img_path = os.path.join(images_dir, f"page_{page_num}.jpg")
        image.save(img_path, "JPEG", quality=80)
        
        ids.append(f"page_{page_num}")
        metadatas.append({"page": page_num, "image_path": img_path})
        texts.append(f"Page {page_num} screenshot content from document {pdf_name}")

    # Add text descriptions to ChromaDB (Embeddings API calls OpenRouter automatically)
    vectorstore.add_texts(
        ids=ids,
        texts=texts,
        metadatas=metadatas
    )
    
    # Cleanup memory
    del images
    gc.collect()
    
    return vectorstore, len(ids)


# Sidebar Credentials Setup
st.sidebar.header("Configuration")
openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")
if not openrouter_api_key:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
if not gemini_api_key:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Main Interface
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file and openrouter_api_key and gemini_api_key:
    # Reset vectorstore state if a new file is uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        if "vectorstore" in st.session_state:
            del st.session_state["vectorstore"]

    # Indexing PDF
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF screenshots & indexing embeddings via OpenRouter (NVIDIA Nemotron)..."):
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            vectorstore, total_pages = sync_init_and_process(
                temp_path, uploaded_file.name, openrouter_api_key
            )
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
            
            # Step 1: Similarity Search via OpenRouter Embeddings
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

elif uploaded_file and (not openrouter_api_key or not gemini_api_key):
    st.info("Please enter both OpenRouter API Key and Gemini API Key in the sidebar or Streamlit Secrets to proceed.")
