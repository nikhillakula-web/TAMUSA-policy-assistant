import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

POLICY_FOLDER = "policies"

documents = []

print(" Loading policy documents...")

for file in os.listdir(POLICY_FOLDER):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(POLICY_FOLDER, file))
        documents.extend(loader.load())

print(f" Loaded {len(documents)} pages")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")

# Create embeddings (LOCAL model)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save locally
vectorstore.save_local("policy_index")

print(" Policy index created successfully!")
