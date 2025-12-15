🧠 byteQgini — Offline Document Intelligence Engine (Fast Local RAG)
====================================================================

![](https://img.shields.io/badge/Build-Stable-00C853?style=for-the-badge) ![](https://img.shields.io/badge/Version-1.1.0-2962FF?style=for-the-badge) ![](https://img.shields.io/badge/Runtime-CPU_Only-blue?style=for-the-badge) ![](https://img.shields.io/badge/Vector_DB-FAISS_HNSW-orange?style=for-the-badge) ![](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-purple?style=for-the-badge) ![](https://img.shields.io/badge/LLM-llama.cpp_GGUF-red?style=for-the-badge) ![](https://img.shields.io/badge/Privacy-100%_Local-green?style=for-the-badge) ![](https://img.shields.io/badge/License-Open_Source-green?style=for-the-badge)

🚀 Overview
-----------

**byteQgini** is a **fully offline, privacy-first Retrieval-Augmented Generation (RAG) system** designed for **fast, deterministic document intelligence on CPU**.

It ingests PDF documents, builds a persistent semantic index, and answers user queries **strictly from the provided documents** — all **without cloud APIs, external services, or data leakage**.

The system is optimized to achieve **~1–3 second response latency on CPU** and is designed to be **downloadable, open-source, and production-ready**.

🎯 Key Design Goals
-------------------

*   🔒 **100% Local & Private** — no external APIs
    
*   ⚡ **Low-latency (1–3s SLA)** on CPU
    
*   🧠 **Strict document grounding** (no hallucinations)
    
*   📦 **Downloadable / sellable offline bundle**
    
*   🛠️ **Deterministic & stable architecture**
    
*   📈 **Scales with growing document sets**
    

🧩 What byteQgini Delivers
--------------------------

*   Automatic ingestion of PDFs from the ./data/ directory
    
*   Deterministic chunking of documents using RecursiveCharacterTextSplitter
    
*   High-quality semantic embeddings via sentence-transformers/all-MiniLM-L6-v2
    
*   **FAISS HNSW** index for fast approximate nearest-neighbor search
    
*   Persistent storage of embeddings and document chunks
    
*   Retrieval-augmented response generation using **llama.cpp (GGUF models)**
    
*   **Static dataset memory + KV-cache reuse** for low latency
    
*   Lightweight Flask HTTP interface for UI or programmatic access
    

🧠 High-Level Architecture
--------------------------
   PDFs (static)     ↓  Chunking + Embeddings (once)     ↓  FAISS HNSW (fast ANN search)     ↓  Relevant document chunk     ↓  llama.cpp (cached system + dataset memory)     ↓  Answer (offline, private, fast)   `

⚙️ How It Works
---------------

### 1️⃣ Startup Initialization

*   Loads existing FAISS index and document store if available
    
*   Otherwise:
    
    *   Reads all PDFs from ./data/
        
    *   Chunks and embeds text
        
    *   Builds a FAISS HNSW index
        
*   Builds a **static dataset summary once**
    
*   Pre-warms the LLM to initialize the **KV cache**
    

### 2️⃣ Retrieval (Fast Path)

*   User query is embedded once
    
*   FAISS HNSW retrieves the most relevant chunk (k = 1)
    
*   Out-of-scope queries are hard-blocked
    

### 3️⃣ Answer Generation (Optimized)

*   Static prompt prefix (system rules + dataset summary) is **cached**
    
*   Only the **context chunk + question** are processed per request
    
*   llama.cpp generates a concise, factual answer
    
*   Typical latency: **~1–3 seconds on CPU**
    

📁 Project Structure
--------------------

`   byteQgini/  ├─ app.py                    # Core application (Flask + RAG engine)  ├─ data/                     # Place your PDF files here  ├─ models/  │   └─ llama.gguf            # GGUF model for llama.cpp  ├─ precomputed_data/  │   ├─ index.faiss           # FAISS HNSW index  │   └─ docs.pkl              # Serialized document chunks  ├─ templates/  │   └─ index.html            # Chat UI  ├─ requirements.txt  └─ README.md   `

🔌 API Endpoints
----------------

### GET /

*   Serves the chat UI (index.html)
    

### GET /get?msg=your+question

*   Returns a plain-text response grounded in documents
    

`   curl "http://127.0.0.1:5000/get?msg=what is covered in these documents"   `

### POST /get

*   Accepts msg as form data
    
*   Same behavior as GET
    

🧪 Installation & Setup
-----------------------

### 1️⃣ Clone Repository

`   git clone https://github.com/byteQ-services/byteQgini.git  cd byteQgini   `

### 2️⃣ Create Virtual Environment

`   python3.11 -m venv .venv  source .venv/bin/activate   `

### 3️⃣ Install Dependencies

`   pip install -r requirements.txt   `

### 4️⃣ Download GGUF Model

`   mkdir -p models  wget -O models/llama.gguf \  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf   `

### 5️⃣ Run Application

`   python app.py   `

Open:

`   http://127.0.0.1:5000   `

🧰 Tech Stack
-------------

### Core

*   Python **3.10 – 3.11** (recommended)
    
*   Flask
    
*   FAISS (CPU, HNSW)
    
*   HuggingFace sentence-transformers
    
*   llama.cpp (llama-cpp-python)
    
*   NumPy
    

### Models

*   Embeddings: all-MiniLM-L6-v2
    
*   LLM: GGUF-quantized LLaMA / compatible instruct models
    

⚠️ Known Constraints
--------------------

*   Single-chunk retrieval (k = 1) by design (for latency)
    
*   No authentication or rate limiting (local use only)
    
*   CPU-only inference (GPU not required)
    

🔮 Roadmap
----------

*   Multi-chunk aggregation
    
*   Confidence / similarity scoring
    
*   Response caching
    
*   Desktop & CLI packaging
    
*   SLA benchmarks & telemetry
    
*   Domain-specific bundles (legal, medical, finance)
    

🏆 Why This Project Matters
---------------------------

Most RAG systems:

*   depend on cloud APIs
    
*   have unpredictable latency
    
*   leak data by design
    

**byteQgini** is different:

*   fully offline
    
*   deterministic
    
*   fast
    
*   and genuinely shippable
    

🤝 Contributions
----------------

Issues, discussions, and pull requests are welcome.This project is built to be **extended, audited, and trusted**.
