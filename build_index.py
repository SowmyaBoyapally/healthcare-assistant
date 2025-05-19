from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docs_dir = "data/medical_docs/"
index_path = "data/embeddings/faiss_index"

all_docs = []
for file in os.listdir(docs_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(docs_dir, file))
        all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

vector_db = FAISS.from_documents(chunks, embedding)
vector_db.save_local(index_path)
