import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai

from dotenv import load_dotenv
load_dotenv()


# Web Title

st.title("RAG: Q&A with Groq and Open Source LLM models")

st.text("Use different model if rate limit exhausts!!!!!")
st.text("Document Embeddings take some times")


# Sidebar setting
st.sidebar.title("Settings")

# Sidebar for API key input
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if groq_api_key:
    # Initialize the LLM model
    engine = st.sidebar.selectbox("Select LLM model", ["gemma-7b-it", "gemma2-9b-it","Llama3-8b-8192","Llama3-70b-8192","mixtral-8x7b-32768"])
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name=engine, temperature=temperature)
        st.sidebar.write("Groq client initialized successfully.")
    except Exception as e:
        st.sidebar.write(f"Error initializing Groq client: {e}")
else:
    st.sidebar.write("Please enter your Groq API key.")





# groq_api_key=st.sidebar.text_input("Enter your Groq AI API Key:",type="password")
os.environ["HF_TOKEN"]=st.sidebar.text_input("Enter your Hugging Face API Key:",type="password")

# os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. If context not availabe based on the question, then you are a helpful AI Assistant.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
    
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("Documents")  # Data Ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        
        if not st.session_state.docs:
            st.write("No documents loaded. Please check the file path and contents.")
            return
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        if not st.session_state.final_documents:
            st.write("No documents split. Please check the splitter configuration.")
            return
        
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        except Exception as e:
            st.write(f"Error creating FAISS vector store: {e}")




    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        st.session_state.loader = PyPDFDirectoryLoader("/Users/abdulwasaysiddiqui/Desktop/GenAI/4-RAG Document Q&A/research_papers"), # Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size= 10000, chunk_overlap = 500), # Splitter Definition
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs) # splitting documents
        st.session_state.vectors = FAISS(st.session_state.final_documents, st.session_state.embeddings) # Creating Vector database


user_prompt=st.text_input("Enter your query as per the context.")


## Sidebar for settings
if st.sidebar.button("Document Embedding"):
    create_vector_embedding()
    st.sidebar.write("Vector Database is ready")
## Select the LLM model


import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chains = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()

    response = retriever_chains.invoke({'input': user_prompt})
    print(f"Response time of model {engine} :{time.process_time()-start}")

    st.write(response['answer'])

     ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

