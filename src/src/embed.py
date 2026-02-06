from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def create_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")

    return db
