
import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DOCS_PATH = "docs"
DB_PATH = "db/chroma_db"
COLLECTION_NAME = "company_documents"


def load_documents():
    print("\nLoading documents...")

    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("Docs folder not found!")

    loader = DirectoryLoader(
        path=DOCS_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No text documents found in docs folder!")

    print(f"Total documents loaded: {len(documents)}")

    for doc in documents:
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content length: {len(doc.page_content)}")

    return documents


def split_documents(documents):
    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Chunk size: {text_splitter._chunk_size}")
    print(f"Chunk overlap: 200")
    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_embedding_model():
    print("\nLoading Hugging Face embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Embedding model loaded.")

    return embedding_model


def create_vector_store(chunks, embedding_model):
    print("\nCreating ChromaDB vector store...")

    os.makedirs("db", exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Chunks stored in ChromaDB: {vectorstore._collection.count()}")

    return vectorstore


def main():
    print("=" * 40)
    print("STARTING DOCUMENT INGESTION")
    print("=" * 40)

    # Load documents
    documents = load_documents()

    # Split documents
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No chunks generated!")

    # Load embedding model
    embedding_model = create_embedding_model()

    # Remove old database so updated chunks are stored
    if os.path.exists(DB_PATH):
        print("\nRemoving old ChromaDB database...")
        shutil.rmtree(DB_PATH)

    # Create fresh vector database
    vectorstore = create_vector_store(chunks, embedding_model)

    print("\n" + "=" * 40)
    print("INGESTION COMPLETED SUCCESSFULLY!")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print("=" * 40)

    return vectorstore


if __name__ == "__main__":
    main()
