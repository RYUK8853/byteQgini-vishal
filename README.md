🧠 byteQgini
Local PDF Intelligence • FAISS Retrieval • Ollama Reasoning
<p align="center"> <img src="https://img.shields.io/badge/Build-Passing-00C853?style=for-the-badge"> <img src="https://img.shields.io/badge/Version-1.0.0-2962FF?style=for-the-badge"> <img src="https://img.shields.io/badge/Framework-Flask-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/Vector_DB-FAISS-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-purple?style=for-the-badge"> <img src="https://img.shields.io/badge/LLM-Ollama_llama3.2-red?style=for-the-badge"> <img src="https://img.shields.io/badge/License-OpenSource-green?style=for-the-badge"> </p>
<p align="center"></p>

# 🧠 byteQgini – Local Document Intelligence and Contextual Response Engine

`byteQgini` is a fully local Retrieval-Augmented Generation (RAG) system that:

- ingests and interprets PDF documents supplied in your designated data directory,
- constructs a persistent FAISS vector index powered by HuggingFace transformer embeddings,
- leverages Ollama (model: `llama3.2`) for contextual natural language response generation,
- exposes a lightweight Flask-based HTTP interface suitable for browser clients, custom frontends, or programmatic access.

The application operates entirely on-device (with the exception of one-time model acquisition), providing ongoing automatic discovery of newly added PDF files and incremental index updates without requiring full reprocessing of the corpus.

## What This Project Delivers (Current Functional Scope)

The system currently provides end-to-end local document intelligence with the following capabilities:

- Automated assimilation of newly provided PDF documents into the system’s local knowledge base via the `./data/` directory
- Deterministic segmentation of document text into semantically aligned chunks using `RecursiveCharacterTextSplitter`
- Embedding of all text segments with `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings`, ensuring efficient vector representation
- Construction and maintenance of a FAISS-based similarity index for high-speed semantic retrieval
- Persistent caching of all processed artifacts, including:
  - `precomputed_data/index.faiss` — FAISS vector index
  - `precomputed_data/docs.pkl` — serialized LangChain `Document` objects
  - `precomputed_data/processed_files.pkl` — registry of PDFs already embedded and indexed
- Continuous ingestion of new content:
  - automatic detection of newly added PDFs both at startup and in scheduled background execution
  - incremental index updates without full reprocessing
- Retrieval-augmented response generation:
  1. FAISS returns the most contextually relevant document segment
  2. `OllamaLLM(model="llama3.2")` synthesizes a refined natural-language answer grounded in retrieved context
- Basic conversational handling for user salutations and exit expressions
- Public HTTP interface enabling interactive and programmatic access:
  - `/` — serves the chat UI defined in `index.html`
  - `/get` — processes user queries supplied via the `msg` parameter and returns generated responses

## Tech Stack

### Backend

- Python `3.10.x – 3.11.x` (recommended; avoid 3.12 for LangChain stability)
- Flask `3.0.2` – web framework
- FAISS CPU `1.8.0.post1` – vector similarity search
- LangChain `0.3.5` (stable compatibility layer) including:
  - `PyPDFLoader` (PDF ingestion)
  - `RecursiveCharacterTextSplitter` (document chunking)
  - `Document` schema for text storage
  - `InMemoryDocstore` for vector-to-document mapping
- HuggingFace Embeddings (`langchain-huggingface==0.0.21`)
  - Model: `sentence-transformers/all-MiniLM-L6-v2` (release 2023 build)
- Ollama Runtime `>=0.1.29`
  - Interface: `OllamaLLM` (`langchain-ollama==0.1.3`)
  - Local Model: `llama3.2` (2024 stable build)

### Supporting Libraries

- numpy `1.26.4` – required for FAISS vector array operations
- pickle (Python stdlib) – persistence of docs and processed filenames
- threading, time, os (Python stdlib) – background scan scheduler and file system management
- pypdf `4.1.0` – PDF parsing backend leveraged by `PyPDFLoader`


## 📁 Folder & File Layout

Expected structure:

```text
byteQgini/
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
```

## ⚙️ How It Works (High-Level Flow)

### 1. App startup

- `initialize()` is called.
- It checks `./data/` for PDF files.
- If `precomputed_data/index.faiss` and `precomputed_data/docs.pkl` exist → tries to load them.
- If loading fails or files are missing → regenerates everything from scratch.

---

### 2. Initial index creation – `regenerate_data()`

When a full rebuild is needed:

1. Loops through all PDFs in `./data/`.
2. Uses `PyPDFLoader` to load pages from each PDF.
3. Concatenates page text and splits it into chunks using `RecursiveCharacterTextSplitter`.
4. Wraps each chunk into a `Document` object.
5. Embeds each chunk using `HuggingFaceEmbeddings`.
6. Creates a FAISS index and adds all embeddings.
7. Saves precomputed data:
   - `docs.pkl` → list of `Document` chunks
   - `processed_files.pkl` → list of file names that were indexed
   - `index.faiss` → FAISS index on disk

---

### 3. New file detection – `process_new_files()`

For incremental updates without full regeneration:

1. Loads `processed_files.pkl` to know which PDFs are already indexed.
2. Scans `./data/` for any new `.pdf` files **not** in that list.
3. For each new PDF:
   - Loads pages with `PyPDFLoader`.
   - Creates `Document` objects.
   - Embeds them and **adds directly** to the existing FAISS index.
4. Updates:
   - Global `docs` list.
   - `InMemoryDocstore`.
   - `index_to_docstore_id` mapping.
5. Saves updated:
   - `docs.pkl`
   - `processed_files.pkl`
   - `index.faiss`

---

### 4. Periodic background check – `periodic_check()`

- Runs in a **daemon thread**.
- Calls `process_new_files()` in an infinite loop.
- Sleeps `time.sleep(1440)` seconds between checks.
  - Note: `1440` seconds ≈ 24 minutes (the code comment says “24 hours”; this can be adjusted later).

---

### 5. Handling user queries – `/get` endpoint

1. Reads the user message (`msg`) from:
   - POST body (`request.form`)
   - or query parameter (`request.args`)
2. If the message is a **greeting** → returns a friendly greeting.
3. If the message is a **farewell** → returns a goodbye message.
4. Otherwise:
   - Calls `search_faiss_index(library, question)`:
     - Embeds the question using `HuggingFaceEmbeddings`.
     - Searches FAISS for the most similar document chunk (`k = 1`).
     - Returns the corresponding text chunk.
   - Calls `refine_answer_with_ollama(ollama_model, context, question)`:
     - Creates a system-style prompt containing:
       - **Context**: the retrieved chunk.
       - **Question**: the user’s question.
     - Sends this prompt to `OllamaLLM(model="llama3.2")`.
     - Returns the generated answer string.

---

## 🔌 API Endpoints

### `GET /`

- Renders the `templates/index.html` file.
- Typically used as the main chat UI page.

---

### `GET /get?msg=your+question`

- **Method:** `GET`
- **Query parameter:** `msg` – the user’s message.
- **Returns:** plain text response from the chatbot.

**Example:**

```bash
curl "http://127.0.0.1:5000/get?msg=what%20is%20in%20these%20documents"
```
## POST /get

**Method:** POST  
**Form field:** `msg` – the user’s message  
**Content-Type expected:** `application/x-www-form-urlencoded`  
**Response:** same as GET

**Example:**
```bash
curl -X POST -d "msg=hello" http://127.0.0.1:5000/get
```
Greetings and Farewells
Hard-coded lists used for basic conversational handling:

python
```cmd
greetings = ["hi", "hello", "hey", "namaste", "hola"]
farewells = ["bye", "goodbye", "see you", "farewell", "no"]
```

If the user input (lowercased) matches any greeting, response is:

```css
Hello! How can I assist you today?
If it matches any farewell, response is:

Goodbye! Have a great day!
```

## Requirements

The project should use the following pinned dependency versions in `requirements.txt`:

flask==3.0.2
faiss-cpu==1.8.0.post1
numpy==1.26.4
langchain-core==0.3.5
langchain-community==0.3.5
langchain-text-splitters==0.0.3
langchain-huggingface==0.0.21
langchain-ollama==0.1.3
sentence-transformers==2.2.2
pypdf==4.1.0
torch==2.1.2


### Notes

- `sentence-transformers` implicitly requires **PyTorch (`torch`)**, which may not auto-install on all environments, so the pinned version is included explicitly.
- `pypdf` is used indirectly by **`PyPDFLoader`** during PDF ingestion and page extraction.
- Python 3.10.x–3.11.x is recommended for full compatibility with:
  - FAISS 1.8.x
  - LangChain 0.3.x
  - HuggingFace stack
- Python 3.12 is not recommended at this time due to partial LangChain compatibility constraints.

---

### Optional GPU Upgrade (If Needed)

If GPU acceleration is required, replace `faiss-cpu` with:

faiss-gpu==1.8.0


and install CUDA-appropriate PyTorch:
Install dependencies:

```bash
       pip install -r requirements.txt
```
Installation and Setup

```1. Clone the repository
        git clone https://github.com/byteQ-services/byteQgini.git
        cd byteQgini
```
```2. Create and activate a virtual environment
        python3 -m venv .venv
        source .venv/bin/activate      # Linux / macOS
        # .venv\Scripts\activate       # Windows
```
```3. Install dependencies
        pip install -r requirements.txt
 ```
```4. Create required directories
         mkdir -p data
         mkdir -p precomputed_data
         mkdir -p templates
```
Place all PDF files inside data/

Ensure templates/index.html exists and posts queries to /get

```5. Install and configure Ollama
         Install Ollama from official source, then pull model:

```
```ollama pull llama3.2
         Run the model to ensure availability:

```
```ollama run llama3.2
         Code reference requires exact model name:
```
ollama_model = OllamaLLM(model='llama3.2')
Running the Application
From the repository root (virtual environment active):

python app.py
```Default server address:
         http://127.0.0.1:5000/
```
Opening / uses the chat interface via index.html

/get receives and processes user messages

## Known Limitations

- Single-chunk retrieval (`k = 1`) may limit contextual depth.
- Embedding model instance is re-created per query rather than persisted.
- Background scan interval is approximately 24 minutes, although labeled as 24 hours.
- If the `data/` directory is empty, initial regeneration may not process any content.
- No authentication or rate limiting; not suitable for public deployment.
- No CI/CD pipeline or automated testing currently implemented.

---

## Planned Enhancements

### Retrieval Improvements
- Transition from `k = 1` to multi-chunk aggregation.
- Add filename and page number metadata to retrieved responses.

### Model Prompting
- Introduce user conversation memory.
- Add structured system role and formatting rules.

### Frontend Upgrades
- Develop modern UI using React.
- Add typing animation and chat bubble styling.

---

## Configuration

- Introduce `.env` configuration file to manage:
  - model name
  - FAISS update interval
  - embedding model configuration
  - storage paths

---

## Monitoring

- Add structured logging for:
  - processing duration
  - response latency
  - retrieval accuracy
  - system uptime

---

## Deployment Upgrades

- Add Docker support and a `docker-compose.yml` for ease of setup.
- Support local, dev, and production environments.

---

## Multi-Model Flexibility

- Allow switching between local LLMs and embedding backends.
- Add configuration selectors for:
  - different embedding models
  - alternative Ollama models
  - quantized vs full-size models

---

## Author / Maintainers

**Project:** byteQgini  
**Organization:** byteQ-services  
**Purpose:** Local retrieval-augmented intelligence using FAISS, HuggingFace embeddings, and Ollama (`llama3.2`).

---

## Contributions

Contributions, feature proposals, and issue reports are welcome via GitHub.
