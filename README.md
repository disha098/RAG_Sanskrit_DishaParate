# 🕉️ Sanskrit RAG Assistant (CPU-Based)

A **local Retrieval-Augmented Generation (RAG) system for Sanskrit** that enables semantic search and contextual question answering over Sanskrit prose texts.  
The system runs **entirely on CPU**, uses **FAISS for retrieval**, a **local LLM for generation**, and provides an interactive **Streamlit UI**.

---

## 🎯 Project Motivation

Natural Language Processing resources for Sanskrit are limited, and training large models from scratch is impractical due to data scarcity.  
This project demonstrates how **Retrieval-Augmented Generation (RAG)** can be effectively used to build a **domain-specific Sanskrit QA system** without model training, while preserving textual authenticity and reducing hallucinations.

This project is designed for:
- Sanskrit text understanding
- Classical literature exploration
- Low-resource language NLP research
- Practical RAG system demonstration

---

## ✨ Features

- 📜 Ingest Sanskrit documents (`.docx`)
- 🔍 Semantic retrieval using FAISS
- 🧠 Context-aware answer generation using a CPU-based LLM
- 🕉️ Supports Sanskrit (Devanagari), transliteration, and English
- 🖥️ Simple and interactive Streamlit UI
- ⚙️ Fully local, no cloud APIs or GPU required

---

## ⚙️ Prerequisites

- Python **3.9 – 3.11**
- CPU-based system (no GPU required)
- 8 GB RAM recommended
- Internet (only for initial model download)

---

## Execution steps

Step 1: Activate Virtual Environment
Make sure your virtual environment is activated.
Windows
venv\Scripts\activate

Step 2: Verify Required Files
Ensure the following files and folders exist:
data/Rag-docs.docx
models/mistral-7b-instruct.Q4_K_M.gguf
ingest.py
app.py
The FAISS index folder (vectorstore/) will be created automatically.

Step 3: Run Document Ingestion (One-Time)
This step processes the Sanskrit dataset and builds the FAISS index.
python ingest.py
Expected output:
✅ Sanskrit DOCX dataset indexed successfully.

Step 4: Launch the Streamlit Application
Start the web interface:
streamlit run app.py
Terminal output:
Local URL: http://localhost:8501
Open the URL in your browser.

Step 5: Interact with the Application
Enter queries in:
-Sanskrit (देवनागरी)
-Transliteration (e.g., karma yoga)
-English
Click Ask to receive answers
-Expand the Retrieved Context section to view source text

