import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import streamlit as st
import requests


# Set API key
api_key = "GOOGLE_API"
if not api_key:
    st.error("Google API Key not found. Please set it in your environment variables.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = "ADD"


# Load and process the PDF
doc_path = "IICI_Chatbot_Requirement_updated_2.pdf"
loader = PyPDFLoader(doc_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(pages)

# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vector_store = FAISS.from_documents(chunks, embedding=embeddings)
vector_store.save_local("faiss_index")

# Load vector store
new_vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = new_vector_store.as_retriever()

# Initialize the LLM and memory
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the system prompt
system_prompt = (
    "You are an AI Assistant that helps users find information in a given context.\n"
    "Provide response in Urdu or English based on user input.\n"
    "Please provide accurate responses based only on the provided context.\n"
    "Limit your response to one or two line if possible."
)

# Create the conversational chain
conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=False,
)

# Define Streamlit app
st.title("IICI Medical Chatbot 👋")

# Maintain chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and chatbot response
if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send the prompt to the conversational chain
    response = conversation.run(prompt)

    # Display and save the assistant's response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
