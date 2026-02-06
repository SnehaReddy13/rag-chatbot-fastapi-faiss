from fastapi import FastAPI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

app = FastAPI()


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load existing FAISS index
db = FAISS.load_local("faiss_index", embeddings)

llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)


@app.get("/ask")
def ask(q: str):
    answer = qa_chain.run(q)
    return {"answer": answer}
