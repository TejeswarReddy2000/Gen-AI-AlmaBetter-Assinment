# 📚 Research Paper Management & Analysis Intelligence System

A **RAG-based Research Paper Assistant** that allows users to upload research PDFs, index them into a vector database, and ask natural language questions to get accurate, citation-aware answers using **LLMs + Embeddings + Vector Search**.

This project is designed to reflect **real-world academic research intelligence platforms** used by universities, R&D labs, and research organizations.

---

## 🚀 Features

* 📄 Upload research paper PDFs (up to 200MB)
* ✂️ Automatic text extraction & intelligent chunking
* 🔢 Vector embedding & indexing
* 🔍 Semantic search over paper content
* 🤖 Question-answering using Groq-hosted LLMs
* ⚡ Fast inference and accurate contextual answers
* 🧠 RAG (Retrieval-Augmented Generation) pipeline

---

## 🏗️ System Architecture (High Level)

1. **PDF Upload**
2. **Text Extraction**
3. **Text Chunking + Overlap**
4. **Embedding Generation**
5. **Vector Store Indexing**
6. **User Query**
7. **Relevant Chunk Retrieval**
8. **LLM Answer Generation (with context)**

---

## 🛠️ Tech Stack

| Layer       | Technology                         |
| ----------- | ---------------------------------- |
| Language    | Python                             |
| UI          | Streamlit                          |
| LLM         | Groq (LLaMA / Mixtral models)      |
| Framework   | LangChain                          |
| Embeddings  | Hugging Face Sentence Transformers |
| Vector DB   | FAISS                              |
| PDF Parsing | PyPDF / PDFPlumber                 |
| Environment | Python 3.10+                       |

---

## 📁 Project Structure

```
research_intelligence/
│
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
├── data/
│   └── uploaded_pdfs/          # Stored PDFs
├── vectorstore/
│   └── faiss_index/            # FAISS vector index
└── utils/
    ├── pdf_loader.py           # PDF text extraction
    ├── text_splitter.py        # Chunking logic
    ├── embeddings.py           # Embedding model
    └── qa_chain.py             # RAG QA pipeline
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/research-paper-intelligence.git
cd research-paper-intelligence
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 🧪 How It Works (RAG Flow)

### Step 1: PDF Processing

* PDF text is extracted
* Text is cleaned and normalized

### Step 2: Chunking

* Text is split into chunks (e.g., 500–1000 characters)
* Chunk overlap ensures context continuity

### Step 3: Embeddings

* Each chunk is converted into a vector using Hugging Face models

### Step 4: Vector Storage

* Embeddings are stored in FAISS for fast similarity search

### Step 5: Query Handling

* User question is embedded
* Similar chunks are retrieved from FAISS

### Step 6: Answer Generation

* Retrieved context + user query sent to LLM
* Final grounded answer is returned

---

## 🔍 Example Queries

* What problem does this paper solve?
* What methodology is used?
* What dataset is mentioned?
* What are the key contributions?
* Summarize the conclusion

---

## ⚠️ Common Issues & Fixes

### ❌ Model Decommissioned Error (Groq)

**Error:**

```
The model `llama3-8b-8192` has been decommissioned
```

**Fix:** Use a supported Groq model, for example:

* `llama-3.3-70b-versatile`
* `mixtral-8x7b-32768`

Update your model initialization in code.

---

## 📈 Future Enhancements

* Multi-paper comparison
* Citation highlighting
* Paper summarization
* Topic-wise clustering
* Research trend analysis
* User authentication
* Cloud vector database (Pinecone / Weaviate)

---

## 🎯 Use Cases

* Academic researchers
* PhD & Masters students
* Research analysts
* University libraries
* Think tanks & R&D teams

---

## 👤 Author

**Mutchu Tejeswar Reddy**
AI / GenAI Engineer (Aspirant)
India

---

## 📜 License

This project is for **educational and research purposes**.


