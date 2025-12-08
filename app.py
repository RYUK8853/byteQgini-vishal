import os
import time
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
from langchain_ollama import OllamaLLM
from langchain_community.docstore.in_memory import InMemoryDocstore

app = Flask(__name__)

# Paths to precomputed data
INDEX_PATH = "precomputed_data/index.faiss"
DOCS_PATH = "precomputed_data/docs.pkl"
PROCESSED_FILES_PATH = "precomputed_data/processed_files.pkl"
DATA_FOLDER = "./data/"

# Initialize variables
library = None
docs = []

def initialize():
    global library, docs

    # Check if the data folder is empty
    if not any(file.endswith(".pdf") for file in os.listdir(DATA_FOLDER)):
        print("Data folder is empty. Regenerating from scratch...")
        docs = []
        regenerate_data()
        return

    # Check if precomputed data exists
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        try:
            with open(DOCS_PATH, 'rb') as f:
                docs = pickle.load(f)
            index = faiss.read_index(INDEX_PATH)
            embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            docstore = InMemoryDocstore(dict(enumerate(docs)))
            index_to_docstore_id = {i: i for i in range(len(docs))}
            library = FAISS(embedding_function=embedding_function, 
                            docstore=docstore, 
                            index=index, 
                            index_to_docstore_id=index_to_docstore_id)
            print("Loaded precomputed data successfully.")
        except (EOFError, pickle.UnpicklingError):
            print("Error loading precomputed data. Regenerating...")
            regenerate_data()
    else:
        regenerate_data()

    # Check for new files on initialization
    process_new_files()

def regenerate_data():
    global library, docs
    
    # Load and split all PDF documents in the data folder
    all_docs = []
    processed_files = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_FOLDER, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            all_docs.extend(pages)
            processed_files.append(filename)

    # Concatenate the text from all pages
    raw_text = " ".join([page.page_content for page in all_docs if page.page_content])

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    text_chunks = text_splitter.split_text(raw_text)

    # Create Document objects for each chunk
    docs = [Document(page_content=chunk) for chunk in text_chunks]

    # Initialize HuggingFace embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS index from document embeddings
    index = faiss.IndexFlatL2(embeddings.embed_query(docs[0].page_content).shape[0])
    for doc in docs:
        index.add(np.array([embeddings.embed_query(doc.page_content)]))

    embedding_function = embeddings
    docstore = InMemoryDocstore(dict(enumerate(docs)))
    index_to_docstore_id = {i: i for i in range(len(docs))}
    library = FAISS(embedding_function=embedding_function, 
                    docstore=docstore, 
                    index=index, 
                    index_to_docstore_id=index_to_docstore_id)

    # Save precomputed data for future use
    with open(DOCS_PATH, 'wb') as f:
        pickle.dump(docs, f)
    with open(PROCESSED_FILES_PATH, 'wb') as f:
        pickle.dump(processed_files, f)
    faiss.write_index(library.index, INDEX_PATH)
    print("Precomputed data generated and saved successfully.")

def process_new_files():
    global library, docs

    # Load the list of already processed files
    if os.path.exists(PROCESSED_FILES_PATH):
        with open(PROCESSED_FILES_PATH, 'rb') as f:
            processed_files = pickle.load(f)
    else:
        processed_files = []

    print("Checking for new files...")

    # Initialize HuggingFace embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Track newly added documents
    new_docs = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".pdf") and filename not in processed_files:
            print(f"New file detected: {filename}")
            file_path = os.path.join(DATA_FOLDER, filename)

            # Load and split the PDF file
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            new_docs.extend(pages)
            processed_files.append(filename)

            # Embed and add new documents to the index
            for page in pages:
                if page.page_content:  # Ensure there's valid text content
                    doc = Document(page_content=page.page_content)
                    docs.append(doc)
                    embedding = embeddings.embed_query(doc.page_content)
                    library.index.add(np.array([embedding]))
                    print(f"Embedded and added to index: {page.page_content[:30]}...")

    # Update the docstore and index_to_docstore_id
    if new_docs:
        docstore = InMemoryDocstore(dict(enumerate(docs)))
        index_to_docstore_id = {i: i for i in range(len(docs))}
        library.docstore = docstore
        library.index_to_docstore_id = index_to_docstore_id

        # Save updated docs and processed files
        with open(DOCS_PATH, 'wb') as f:
            pickle.dump(docs, f)
        with open(PROCESSED_FILES_PATH, 'wb') as f:
            pickle.dump(processed_files, f)
        faiss.write_index(library.index, INDEX_PATH)

        print("New files processed and saved successfully.")
    else:
        print("No new files to process.")

# Function to periodically check for new files
def periodic_check():
    while True:
        process_new_files()
        time.sleep(1440)  # Check every 24 hours

# Initialize the system
initialize()

# Start periodic checks in a background thread
thread = threading.Thread(target=periodic_check, daemon=True)
thread.start()

# Initialize Ollama LLM model
ollama_model = OllamaLLM(model='llama3.2')

# List of greetings and farewells
greetings = ["hi", "hello", "hey", "namaste", "hola"]
farewells = ["bye", "goodbye", "see you", "farewell", "no"]

# Function to search FAISS index and return the most relevant text chunk
def search_faiss_index(library, question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    question_embedding = embeddings.embed_query(question)
    D, I = library.index.search(np.array([question_embedding]), 1)  # k=1 for the closest match
    doc_id = I[0][0]
    return library.docstore._dict[doc_id].page_content if doc_id != -1 else None

# Function to refine answer with Ollama LLM
def refine_answer_with_ollama(model, context, question):
    if not context:
        return "I'm sorry, but I couldn't find an answer to your question."
    
    template = """
    You are an expert assistant. Based on the context provided, answer the question concisely and in a more human form. If you think the question is out of context, simply reply that you don't know the answer.

    Context:
    {Context}

    Question: {question}

    Answer:
    """
    input_prompt = template.format(Context=context, question=question)
    result = model.invoke(input=input_prompt)
    return result

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def get_bot_response():
    if request.method == "POST":
        userText = request.form.get('msg').strip().lower()
    else:
        userText = request.args.get('msg').strip().lower()
    
    # Handle greetings
    if userText in greetings:
        return "Hello! How can I assist you today?"
    
    # Handle farewells
    if userText in farewells:
        return "Goodbye! Have a great day!"

    # Handle normal queries
    context = search_faiss_index(library, userText)
    response = refine_answer_with_ollama(ollama_model, context, userText)
    
    return response

if __name__ == "__main__":
    app.run(debug=True)
