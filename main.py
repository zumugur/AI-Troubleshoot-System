# import os
# import streamlit as st
# import time
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv('GROQ_API_KEY')

# st.set_page_config(page_title="Gemma Doc Q&A", layout="wide")
# st.title("📄 Gemma Model Document Q&A")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template(
# """
# You are an advanced AI assistant specializing in document analysis. You will provide detailed, context-aware, and elaborative responses based on the retrieved document content.

# Context:
# <context>
# {context}
# <context>

# User Question:
# {input}

# Provide a detailed, well-explained, and long-form response based on the provided context.
# """
# )

# def load_and_embed_documents():
#     """Loads and embeds PDFs on startup."""
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         all_docs = []
#         pdf_directory = "./input_sample"

#         if not os.path.exists(pdf_directory) or not os.listdir(pdf_directory):
#             st.error("❌ No PDFs found in the input_sample directory.")
#             return

#         with st.spinner("🔄 Embedding PDFs... Please wait."):
#             for filename in os.listdir(pdf_directory):
#                 if filename.endswith(".pdf"):
#                     file_path = os.path.join(pdf_directory, filename)
#                     loader = PyMuPDFLoader(file_path)
#                     docs = loader.load()
#                     all_docs.extend(docs)

#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             final_documents = text_splitter.split_documents(all_docs)

#             st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
#             st.success("✅ PDFs embedded and ready for Q&A!")

# # Embed once on load
# load_and_embed_documents()

# # Input box
# user_input = st.text_input("💬 Ask a question about the documents")

# if user_input:
#     if "vectors" in st.session_state:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         with st.spinner("🤖 Generating answer..."):
#             start = time.process_time()
#             response = retrieval_chain.invoke({'input': user_input})
#             st.write(response['answer'])
#             st.caption(f"⏱️ Answer generated in {round(time.process_time() - start, 2)}s")

#         with st.expander("📚 Document Similarity Search"):
#             for doc in response["context"]:
#                 st.write(doc.page_content)
#                 st.write("———")
#     else:
#         st.error("Embedding failed. Please check your PDFs or restart the app.")










import os
import streamlit as st
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Set page config
st.set_page_config(page_title="AI Troubleshooting System PoC", layout="wide", page_icon="🔧")

# Center the Streamlit content with custom CSS
st.markdown("""
    <style>
    .block-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding-top: 2rem;
    }
    .stTextInput>div>div>input {
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title and Subtitle
st.title("🧠AI Troubleshooting Assistance")
st.subheader("Get accurate troubleshooting instructions based on locomotive data and technical documentation.")

# Load model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt for the AI
prompt = ChatPromptTemplate.from_template("""
You are an advanced AI assistant specializing in locomotive troubleshooting. Based on the provided technical documentation, you will generate context-aware, accurate, and detailed troubleshooting instructions for the issue described by the user.

Context:
<context>
{context}
<context>

User Question:
{input}

Provide a well-structured and elaborate response, using only the information retrieved from the documents.
""")

def load_and_embed_documents():
    """Loads and embeds PDFs on startup."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        all_docs = []
        pdf_directory = "./input_sample"

        if not os.path.exists(pdf_directory) or not os.listdir(pdf_directory):
            st.error("❌ No PDFs found in the input_sample directory.")
            return

        with st.spinner("🔄 Embedding PDFs... Please wait."):
            for filename in os.listdir(pdf_directory):
                if filename.endswith(".pdf"):
                    file_path = os.path.join(pdf_directory, filename)
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    all_docs.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(all_docs)

            st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
            st.success("✅ System ready for troubleshooting!")

# Embed documents once on load
load_and_embed_documents()

# Input box for user query
user_input = st.text_input("💬 Enter your problem description or issue (e.g., 'Locomotive DE24333 has electrical issue with control system')")

if user_input:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("🤖 Retrieving data... Please wait."):
            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_input})
            st.write(response['answer'])
            st.caption(f"⏱️ Answer generated in {round(time.process_time() - start, 2)}s")

        # Optional: Check for alarm or data acquisition system correlation
        st.expander("📊 Check Alarm Data:")  
        # Here, implement logic to check the alarm page or data acquisition system
        # Example: You can link the plate number from the query to specific data from a database.

        # Example of displaying retrieved document context (if needed)
        with st.expander("📚 Document Context:"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("———")
    else:
        st.error("❌ Embedding failed. Please check your PDFs or restart the app.")
