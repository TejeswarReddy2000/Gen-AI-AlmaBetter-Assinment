from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_vector_store(chunks):
    return FAISS.from_texts(chunks, embedding)
