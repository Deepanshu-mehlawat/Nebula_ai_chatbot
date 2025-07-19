# File: app.py

import os
import streamlit as st
from dotenv import load_dotenv

# --- CORRECTED IMPORTS ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA
# --- END OF CORRECTIONS ---

# Load environment variables from .env
load_dotenv()

# App Configuration
st.set_page_config(page_title="Enterprise Document Q&A", layout="wide")
st.title("📄 Enterprise Document Q&A Chatbot")
st.write("Ask questions about the content of your uploaded documents.")

# Load API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in the .env file.")
    st.stop()

# Load Persistent Vector Store
PERSIST_DIRECTORY = "./chroma_db"
if not os.path.exists(PERSIST_DIRECTORY):
    st.error("Vector database not found. Please run the process.py script first.")
    st.stop()

@st.cache_resource
def load_llm_and_vectorstore():
    """Load the LLM and vector store only once."""
    # --- CORRECTED INITIALIZATIONS ---
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )
    
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    # --- END OF CORRECTIONS ---
    
    return llm, vector_db

try:
    llm, vector_db = load_llm_and_vectorstore()

    # Create the RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 results
        return_source_documents=True
    )

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa_chain({"query": prompt})
                response_text = result["result"]
                
                sources = {doc.metadata.get("source", "Unknown") for doc in result["source_documents"]}
                source_text = "\n\n*Sources:* " + ", ".join(sources)
                
                full_response = response_text + source_text
                st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

except Exception as e:
    st.error(f"An error occurred: {e}")
