from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def create_qa(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=vectorstore.as_retriever()
    )
