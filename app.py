import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from unidecode import unidecode

# ---------------- CONFIG ----------------
VECTOR_DB_DIR = "vectorstore"
MODEL_PATH = "models/mistral-7b-instruct.Q4_K_M.gguf"

st.set_page_config(
    page_title="Sanskrit RAG Assistant",
    page_icon="🕉️",
    layout="wide"
)

# ---------------- LOAD COMPONENTS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

@st.cache_resource
def load_vectorstore(embeddings):
    return FAISS.load_local(VECTOR_DB_DIR, embeddings)

@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.2,
        max_tokens=512,
        n_ctx=2048,
        verbose=False
    )

embeddings = load_embeddings()
db = load_vectorstore(embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = load_llm()

# ---------------- UI ----------------
st.title("🕉️ Sanskrit RAG Assistant")
st.caption("Ask questions on Sanskrit stories using Retrieval-Augmented Generation")

st.markdown("""
**Supported Inputs**
- Sanskrit (देवनागरी)
- Transliteration (karma yoga, murkha bhritya)
- English
""")

query = st.text_area(
    "Enter your question:",
    placeholder="मूर्खभृत्यस्य कथायाः नीति किम् अस्ति?",
    height=100
)

submit = st.button("🔍 Ask")

# ---------------- LOGIC ----------------
def normalize_query(q):
    return q + "\n" + unidecode(q)

if submit and query.strip():
    with st.spinner("Retrieving Sanskrit context..."):
        query_norm = normalize_query(query)
        docs = retriever.get_relevant_documents(query_norm)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are a Sanskrit scholar.

Context (from Sanskrit prose stories):
{context}

Question:
{query}

Answer clearly.
Explain the story and moral if applicable.
Use simple Sanskrit or Hindi.
"""

    with st.spinner("Generating answer..."):
        response = llm(prompt)

    st.subheader("📜 Answer")
    st.write(response)

    with st.expander("📚 Retrieved Sanskrit Context"):
        st.write(context)

else:
    st.info("Enter a question and click **Ask** to begin.")
