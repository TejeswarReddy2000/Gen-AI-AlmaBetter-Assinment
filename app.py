# =============================
# LOAD ENV VARIABLES
# =============================
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

# =============================
# STREAMLIT CHECK
# =============================
import streamlit as st

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# =============================
# IMPORTS
# =============================
import fitz  # PyMuPDF

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Research Paper Intelligence System",
    layout="wide"
)

st.title("📚 Research Paper Management & Analysis Intelligence System")
st.write("RAG-based Research Paper Assistant using **Groq LLaMA 3.1**")

# =============================
# CACHED FUNCTIONS
# =============================
@st.cache_data(show_spinner=False)
def parse_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


@st.cache_data(show_spinner=False)
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


def create_qa_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff"
    )

# =============================
# STREAMLIT UI
# =============================
uploaded_file = st.file_uploader(
    "Upload a Research Paper PDF",
    type=["pdf"]
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    pdf_bytes = uploaded_file.read()

    with st.spinner("📄 Extracting text from PDF..."):
        text = parse_pdf(pdf_bytes)

    st.info(f"Extracted text length: {len(text)} characters")

    with st.spinner("✂️ Chunking document..."):
        chunks = chunk_text(text)

    st.info(f"Total chunks created: {len(chunks)}")

    with st.spinner("🔢 Building vector index (first time may take 1–2 minutes)..."):
        vectorstore = build_vector_store(chunks)

    qa_chain = create_qa_chain(vectorstore)

    st.success("✅ Paper indexed successfully!")

    st.subheader("🔍 Ask a question about the paper")
    query = st.text_input("Example: What problem does this paper solve?")

    if query:
        with st.spinner("🤖 Generating answer..."):
            answer = qa_chain.run(query)

        st.subheader("🧠 Answer")
        st.write(answer)

else:
    st.warning("Please upload a PDF to begin.")
