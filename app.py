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

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

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


def encode_image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
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

    ids, embeddings_list, metadatas = [], [], []
    for idx, image in enumerate(images):
        page_num = idx + 1
        img_path = os.path.join(images_dir, f"page_{page_num}.jpg")
        image.save(img_path, "JPEG")
        img_emb = embedding_function.embed_image(img_path)
        
        ids.append(f"page_{page_num}")
        embeddings_list.append(img_emb)
        metadatas.append({"page": page_num, "image_path": img_path})

    vectorstore._collection.add(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas
    )
    return vectorstore, len(images)


# Sidebar Key Entry
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
if not groq_api_key:
    groq_api_key = os.getenv("GROQ_API_KEY")

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file and groq_api_key:
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing PDF screenshots & indexing embeddings..."):
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            vectorstore, total_pages = sync_init_and_process(temp_path, uploaded_file.name)
            st.session_state.vectorstore = vectorstore
            st.success(f"Indexed {total_pages} pages successfully!")

    query = st.text_input("Ask a question about the PDF:")
    if query and "vectorstore" in st.session_state:
        with st.spinner("Retrieving relevant pages & generating response..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
            relevant_docs = retriever.invoke(query)

            content_payload = [
                {"type": "text", "text": f"Answer based strictly on the provided page screenshots.\n\nQuery: {query}"}
            ]

            cols = st.columns(len(relevant_docs))
            for idx, doc in enumerate(relevant_docs):
                img_path = doc.metadata.get("image_path")
                page_num = doc.metadata.get("page")
                if img_path and os.path.exists(img_path):
                    base64_img = encode_image_to_base64(img_path)
                    content_payload.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })
                    with cols[idx]:
                        st.image(img_path, caption=f"Page {page_num}")

            llm_vision = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.1
            )
            response = llm_vision.invoke([HumanMessage(content=content_payload)])
            st.markdown("### Answer:")
            st.write(response.content)
