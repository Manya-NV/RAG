
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Vector database location
persistent_directory = "db/chroma_db"


# Load the same embedding model used during ingestion
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# Load existing vector database
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_name="company_documents"
)


# User query
query = "What does Google specialize in?"


# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# Retrieve relevant documents
relevant_docs = retriever.invoke(query)


# Display results
print(f"\nUser Query: {query}")

print("\n--- Retrieved Context ---")

for i, doc in enumerate(relevant_docs, 1):

    print(f"\nDocument {i}:")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))
