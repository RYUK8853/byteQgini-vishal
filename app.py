import os
import faiss
import numpy as np
import pickle
import threading
from flask import Flask, render_template, request

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from llama_cpp import Llama

# =========================
# CONFIG
# =========================
DATA_FOLDER = "./data/"
INDEX_PATH = "precomputed_data/index.faiss"
DOCS_PATH = "precomputed_data/docs.pkl"

MODEL_PATH = "models/llama.gguf"   # GGUF model path
CTX_SIZE = 4096
MAX_TOKENS = 256                   # 🔑 latency control

os.makedirs("precomputed_data", exist_ok=True)

# =========================
# APP
# =========================
app = Flask(__name__)

# =========================
# GLOBALS
# =========================
library = None
docs = []
index_lock = threading.Lock()

GREETINGS = {"hi", "hello", "hey", "namaste", "hola"}
FAREWELLS = {"bye", "goodbye", "see you", "farewell"}

# =========================
# EMBEDDINGS (ONE INSTANCE)
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# STATIC KNOWLEDGE (STEP 3)
# =========================
def build_dataset_summary(docs, max_chars=1500):
    if not docs:
        return "No documents loaded."

    joined = " ".join(d.page_content for d in docs[:10])
    _ = joined[:max_chars]

    return """
DATASET OVERVIEW:
The uploaded documents are static reference PDFs.
They contain factual information intended for question answering.
The assistant must answer strictly from these documents.
If an answer is not present, say "I don't know".
"""

# =========================
# LOAD LLM (STEP 4)
# =========================
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=os.cpu_count(),
    n_batch=512,
    verbose=False
)

# =========================
# INITIALIZATION
# =========================
def initialize():
    global library, docs

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Load precomputed index if available
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        docs = pickle.load(open(DOCS_PATH, "rb"))
        index = faiss.read_index(INDEX_PATH)
        docstore = InMemoryDocstore(dict(enumerate(docs)))
        library = FAISS(
            embedding_function=embeddings,
            docstore=docstore,
            index=index,
            index_to_docstore_id={i: i for i in range(len(docs))}
        )
        print("Loaded existing FAISS index.")
        return

    regenerate_data()

def regenerate_data():
    global library, docs

    pages = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
            pages.extend(loader.load())

    raw_text = " ".join(p.page_content for p in pages if p.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80
    )

    chunks = splitter.split_text(raw_text)
    docs = [Document(page_content=c) for c in chunks]

    texts = [d.page_content for d in docs]
    vectors = np.array(embeddings.embed_documents(texts)).astype("float32")

    dim = vectors.shape[1]

    # 🔥 STEP 2: HNSW FAISS
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 32
    index.add(vectors)

    docstore = InMemoryDocstore(dict(enumerate(docs)))
    library = FAISS(
        embedding_function=embeddings,
        docstore=docstore,
        index=index,
        index_to_docstore_id={i: i for i in range(len(docs))}
    )

    pickle.dump(docs, open(DOCS_PATH, "wb"))
    faiss.write_index(index, INDEX_PATH)

    print("FAISS index regenerated.")

# =========================
# SEARCH (FAST + GUARDED)
# =========================
def search_faiss_index(question):
    with index_lock:
        q_emb = np.array(embeddings.embed_query(question)).astype("float32")
        D, I = library.index.search(q_emb.reshape(1, -1), 1)

        # Hard out-of-scope guard
        if I[0][0] == -1 or D[0][0] > 1.0:
            return None

        return library.docstore._dict[I[0][0]].page_content

# =========================
# STARTUP SEQUENCE
# =========================
initialize()

# STEP 3: Static dataset memory
DATASET_SUMMARY = build_dataset_summary(docs)

# STEP 5: Static prefix (KV-cache friendly)
STATIC_PREFIX = f"""
You are a private offline assistant.

{DATASET_SUMMARY}

RULES:
- Answer only from the context.
- If the answer is not present, say "I don't know".
- Be concise and factual.
"""

# 🔥 STEP 5: Pre-warm KV cache ONCE
llm(STATIC_PREFIX, max_tokens=1)

# =========================
# ANSWERING (STEP 5)
# =========================
def refine_answer(context, question):
    if not context or len(context.strip()) < 50:
        return "I can only answer questions based on the uploaded documents."

    prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    output = llm(
        STATIC_PREFIX + prompt,
        max_tokens=MAX_TOKENS,
        stop=["\n\n"]
    )

    return output["choices"][0]["text"].strip()

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    q = (request.form.get("msg") or request.args.get("msg") or "").strip()
    if not q:
        return "Ask a question."

    q_lower = q.lower()

    if q_lower in GREETINGS:
        return "Hello! How can I help you today?"
    if q_lower in FAREWELLS:
        return "Goodbye! Have a great day."

    context = search_faiss_index(q)
    if not context:
        return "I can only answer questions based on the uploaded documents."

    return refine_answer(context, q)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
