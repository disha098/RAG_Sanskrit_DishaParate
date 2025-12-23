from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from unidecode import unidecode

VECTOR_DB_DIR = "vectorstore"
MODEL_PATH = "models/mistral-7b-instruct.Q4_K_M.gguf"

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Load vector store
db = FAISS.load_local(VECTOR_DB_DIR, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load CPU LLM
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.2,
    max_tokens=512,
    n_ctx=2048,
    verbose=False
)

def normalize_query(q):
    # Helps transliterated Sanskrit
    return q + "\n" + unidecode(q)

while True:
    query = input("\n🕉️ Ask a Sanskrit question (exit to quit): ")

    if query.lower() == "exit":
        break

    query_norm = normalize_query(query)
    docs = retriever.get_relevant_documents(query_norm)

    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
You are a Sanskrit scholar.

Context (from Sanskrit stories):
{context}

Question:
{query}

Answer clearly.
Explain the story or moral if relevant.
Use simple Sanskrit or Hindi.
"""

    response = llm(prompt)
    print("\n📜 Answer:\n", response)
