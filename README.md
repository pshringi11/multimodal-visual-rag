# Multimodal Visual RAG Agent

A Multimodal Visual Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, **OpenCLIP**, **ChromaDB**, and **Groq Llama 4 Vision**. 

🚀 **Live App**: [https://multimodal-visual-rag.streamlit.app/](https://multimodal-visual-rag.streamlit.app/)

Instead of relying on standard text extractions (which often drop formatting, charts, and diagrams), this agent converts PDF document pages into high-resolution image screenshots. It indexes page visual embeddings using **OpenCLIP** and **ChromaDB**, retrieves relevant page screenshots for a given user query, and feeds them to **Groq's Llama 4 Scout Vision model** to generate accurate, context-aware answers.

---

## 🌟 Key Features

* **Visual PDF Processing**: Renders PDF pages into images using `poppler-utils` and `pdf2image` to preserve full visual context (tables, charts, layout structure).
* **OpenCLIP Embeddings**: Uses `ViT-B-32` (`laion2b_s34b_b79k`) vision-text embeddings to map queries and document images into a shared vector space.
* **Vector Storage with ChromaDB**: Local vector database for efficient similarity search across document page embeddings.
* **Groq Llama 4 Scout Vision**: Integrates `meta-llama/llama-4-scout-17b-16e-instruct` for fast, accurate vision-language reasoning.
* **Interactive Streamlit UI**: User-friendly web interface supporting seamless PDF uploads, query inputs, and side-by-side visual source inspections.
* **Cloud Ready**: Configured for 1-click free deployment on **Streamlit Community Cloud**.

---

## 🏗️ Architecture & Workflow

1. **PDF Ingestion & Conversion**: The uploaded PDF is converted into JPEG screenshots on a page-by-page basis.
2. **Visual Embedding Generation**: Each page image is processed via OpenCLIP (`ViT-B-32`) to produce image feature vectors.
3. **Indexing**: Embeddings along with page metadata (page number, image path) are stored in ChromaDB.
4. **Retrieval**: When a user submits a prompt, OpenCLIP embeds the text query, and ChromaDB retrieves the top matching page screenshots.
5. **Multimodal Generation**: The retrieved page images and user prompt are encoded into a multimodal payload and sent to Groq Llama 4 Scout Vision to return a grounded response.

---

## 🛠️ Tech Stack

* **Frontend / Framework**: Streamlit
* **Vision & Embeddings**: OpenCLIP (`open_clip_torch`), PyTorch, Pillow
* **Vector Database**: ChromaDB (`langchain-chroma`)
* **LLM / Multimodal Orchestration**: LangChain, Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`)
* **System Utilities**: `poppler-utils`, `pdf2image`

---

## 📂 Repository Structure

```text
├── app.py              # Main Streamlit application entry point
├── requirements.txt    # Python package dependencies
├── packages.txt        # System-level dependencies (poppler-utils)
├── .gitignore          # Git ignore patterns
└── README.md           # Project documentation
