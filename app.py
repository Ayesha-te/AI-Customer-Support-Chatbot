import openai
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Set your OpenAI API key (replace with your own API key)
openai.api_key = "your-openai-api-key"

# --- Setup OpenAI Embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# --- Initialize FAISS index ---
# FAISS index dimension should match the embedding dimensions
# OpenAI embeddings (GPT-3 models) typically have 1536 dimensions
index = faiss.IndexFlatL2(1536)

# We create the FAISS vector store using OpenAI embeddings
vectorstore = FAISS(embedding_function=embeddings.embed_query, faiss_index=index)

# --- Setup Memory for Conversation ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Setup the LLM ---
llm = ChatOpenAI(openai_api_key=openai.api_key, temperature=0)

# --- Setup Conversational Retrieval Chain ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# --- Streamlit UI ---
st.title("AI Customer Support Chatbot")

# Input field for the user's query
user_query = st.text_input("Ask me anything:")

if user_query:
    # Get a response from the model using the conversational retrieval chain
    response = qa_chain.run(input=user_query)
    st.write(response)

# Example: Add some documents to the FAISS vector store for testing purposes
# (In a real scenario, you would replace this with dynamic documents or FAQs)
documents = [
    "How do I reset my password?",
    "What is your refund policy?",
    "How can I contact customer support?",
    "Where is my order?"
]

# Embedding and storing documents in FAISS
for doc in documents:
    doc_embedding = embeddings.embed_query(doc)
    vectorstore.add_texts([doc], [doc_embedding])

