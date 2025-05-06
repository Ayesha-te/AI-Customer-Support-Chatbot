import openai
import faiss
import numpy as np
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# --- Set your API keys ---
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# --- LangChain Embeddings & VectorStore ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create a FAISS index (you can modify this based on your use case)
index = faiss.IndexFlatL2(1536)  # Dimension should match your embedding size

# Create the FAISS vector store
vectorstore = FAISS(embedding_function=embeddings.embed_query, faiss_index=index)

# --- Setup Memory for Conversation ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Setup LLM ---
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

# --- Setup Conversational Chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(),
    memory=memory,
)

# --- Streamlit Interface ---
st.title("AI Customer Support Chatbot")

# User input for the chatbot
user_input = st.text_input("Ask me anything:")

if user_input:
    response = qa_chain.run(user_input)
    st.write(response)


