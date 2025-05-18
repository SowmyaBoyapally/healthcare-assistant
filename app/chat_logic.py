from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_path = "data/embeddings/faiss_index"

def load_rag_pipeline():
    vector_store = FAISS.load_local(db_path, embedding)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=vector_store.as_retriever()
    )
    return qa_chain

def get_rag_response(query, qa_chain):
    return qa_chain.run(query)

