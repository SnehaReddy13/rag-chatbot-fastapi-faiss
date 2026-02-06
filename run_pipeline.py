from src.ingest import load_documents
from src.embed import create_vector_db


def main():
    docs = load_documents("data/sample.pdf")
    create_vector_db(docs)
    print("Vector DB created successfully")


if __name__ == "__main__":
    main()
