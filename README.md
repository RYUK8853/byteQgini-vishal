# 🧠 byteQgennie – Local PDF + Ollama Chatbot (FAISS + LangChain)

`byteQgennie` is a local Retrieval-Augmented Generation (RAG) chatbot that:

- reads your **PDF files** from a folder,
- builds a **FAISS vector index** using **HuggingFace embeddings**,
- uses **Ollama (llama3.2)** as the LLM,
- serves a **Flask web endpoint** that you can call from a frontend or simple HTML chat UI.

It is designed to run **fully locally** (aside from model downloads), with automatic detection of new PDFs and periodic index updates.

---

## 🚀 What This Project Does (Current Capabilities)

Right now, this project is capable of:

- 📄 **Loading PDFs** from the `./data/` folder
- ✂️ **Splitting text into chunks** using `RecursiveCharacterTextSplitter`
- 🧬 **Embedding text chunks** using `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`
- 📚 **Building a FAISS index** for fast similarity search
- 💾 **Saving and reusing precomputed data**:
  - `precomputed_data/index.faiss` – FAISS index
  - `precomputed_data/docs.pkl` – list of LangChain `Document` chunks
  - `precomputed_data/processed_files.pkl` – names of PDFs already indexed
- 🔁 **Detecting new PDFs automatically** on startup and in a **background thread** (periodic checks)
- 🤖 **Answering user questions** using:
  1. FAISS to find the most relevant chunk  
  2. `OllamaLLM(model="llama3.2")` to generate a natural, refined answer
- 👋 Simple **greetings & farewells**:
  - Responds nicely to “hi”, “hello”, “bye”, etc.
- 🌐 Exposes endpoints:
  - `/` – renders `index.html` template (simple chat UI)
  - `/get` – returns the chatbot response for a `msg` query

---

## 🧱 Tech Stack

### Backend

- **Python**
- **Flask** – web framework
- **FAISS** – vector similarity search (via `faiss` + `langchain_community.vectorstores.FAISS`)
- **LangChain** – for:
  - `PyPDFLoader` (PDF loading)
  - `RecursiveCharacterTextSplitter` (chunking)
  - `Document` type
  - `InMemoryDocstore`
- **HuggingFaceEmbeddings**
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Ollama LLM**
  - `OllamaLLM` from `langchain_ollama`
  - Model: `llama3.2`

### Supporting Libraries

- `numpy` – for numeric arrays used by FAISS
- `pickle` – for serializing docs + processed file names
- `threading`, `time`, `os` – standard library for background tasks and file management

---

## 📁 Folder & File Layout

Expected structure:

```text
byteQgennie/
├─ app.py                       # (the file you shared)
├─ data/                        # <--- Put your PDF files in here
│   ├─ doc1.pdf
│   ├─ doc2.pdf
│   └─ ...
├─ precomputed_data/            # <--- Generated automatically on first run
│   ├─ index.faiss
│   ├─ docs.pkl
│   └─ processed_files.pkl
├─ templates/
│   └─ index.html               # Flask HTML template for the chat UI
├─ requirements.txt             # Python dependencies (see below)
└─ README.md
