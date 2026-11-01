@"
# Mini RAG Application

A document retrieval application built using Python, LangChain, Hugging Face embeddings, ChromaDB, and Streamlit.

## Features
- Load multiple text documents
- Split documents into smaller chunks
- Generate embeddings using Hugging Face
- Store embeddings in ChromaDB
- Retrieve relevant document chunks using semantic search
- MMR-based retrieval for diverse results

## Tech Stack
- Python
- LangChain
- Hugging Face
- ChromaDB
- Streamlit

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Run ingestion:

    python ingestion.py

Start the application:

    streamlit run app.py
"@ | Set-Content README.md