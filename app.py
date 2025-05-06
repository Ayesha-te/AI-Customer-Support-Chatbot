import openai
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import numpy as np

# --- Set your API keys ---
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

# --- Define FAQ Documents or Knowledge Base ---
faq_documents = [
    {"question": "What are your business hours?", "answer": "Our business hours are from 9 AM to 5 PM, Monday through Friday."},
    {"question": "How can I track my order?", "answer": "You can track your order using the tracking number sent to your email."},
    {"question": "What is the return policy?", "answer": "Our return policy allows returns within 30 days of purchase."}
]

# --- LangChain Embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- Create OpenAI's Embedding for FAQ Documents ---
faq_embeddings = [
    embeddings.embed_query(doc['question']) for doc in faq_documents
]

# --- Setup Memory for Conversation ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Setup LLM ---
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

# --- Conversational Retrieval Chain ---
def get_most_relevant_answer(user_input):
    user_input_embedding = embeddings.embed_query(user_input)
    
    # Find the most similar question
    similarities = [np.dot(user_input_embedding, faq_embedding) for faq_embedding in faq_embeddings]
    best_match_index = np.argmax(similarities)
    
    # Retrieve the answer to the most similar question
    return faq_documents[best_match_index]["answer"]

# --- Streamlit Interface ---
st.title("AI Customer Support Chatbot")

# User input for the chatbot
user_input = st.text_input("Ask me anything:")

if user_input:
    response = get_most_relevant_answer(user_input)
    st.write(response)
