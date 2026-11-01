
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Page configuration
st.set_page_config(
    page_title="Mini RAG Application",
    page_icon="📄"
)

st.title("Document Retrieval Application")

st.write("Ask questions about your uploaded documents.")


# Load embedding model
@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# Load vector database
@st.cache_resource
def load_vector_database():

    embedding_model = load_embedding_model()

    return Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
        collection_name="company_documents"
    )


# User input
query = st.text_input("Enter your question:")


# Search button
if st.button("Search"):

    if query.strip():

        db = load_vector_database()

        # MMR retrieval for relevant and diverse chunks
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        relevant_docs = retriever.invoke(query)

        st.subheader("Retrieved Documents")

        if relevant_docs:

            for i, doc in enumerate(relevant_docs, 1):

                st.write(f"### Document {i}")

                st.write(doc.page_content)

                st.caption(
                    f"Source: {doc.metadata.get('source', 'Unknown')}"
                )

        else:

            st.warning("No relevant documents found.")

    else:

        st.warning("Please enter a question.")
