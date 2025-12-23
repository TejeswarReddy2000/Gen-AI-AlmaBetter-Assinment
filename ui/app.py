import streamlit as st
from ingestion.pdf_parser import parse_pdf
from indexing.chunking import chunk_text
from indexing.vector_store import build_vector_store
from rag.qa_engine import create_qa

st.title("📚 Research Paper Assistant")

text = parse_pdf("data/pdfs/sample.pdf")
chunks = chunk_text(text)
vectorstore = build_vector_store(chunks)
qa = create_qa(vectorstore)

query = st.text_input("Ask a question about the paper")

if query:
    st.write(qa.run(query))
