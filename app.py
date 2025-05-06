import openai
import streamlit as st
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Set up your OpenAI API key (make sure to keep it in the secrets file)
openai_api_key = st.secrets["openai_api_key"]

# --- Setup OpenAI Embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# --- Example FAQ Documents ---
faq_documents = [
    {"question": "What are your business hours?", "answer": "We are open from 9 AM to 5 PM, Monday to Friday."},
    {"question": "What is your return policy?", "answer": "Our return policy lasts 30 days from the purchase date."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship internationally, with extra fees depending on the country."},
    {"question": "How can I contact customer support?", "answer": "You can reach us by email at support@example.com."}
]

# Embedding FAQ questions for vector search
faq_questions = [doc['question'] for doc in faq_documents]
faq_embeddings = [embeddings.embed_query(q) for q in faq_questions]

# --- Simple In-Memory Vector Store --- (No FAISS)
# Storing embeddings and corresponding answers in a dictionary
faq_vector_store = {i: {"question": faq_questions[i], "embedding": faq_embeddings[i], "answer": faq_documents[i]["answer"]} for i in range(len(faq_documents))}

# --- Setup Memory for Conversation ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Setup LangChain LLM ---
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Function to retrieve the most similar FAQ based on the user's question
def retrieve_answer(user_question):
    user_embedding = embeddings.embed_query(user_question)
    similarities = []

    # Calculate similarity (cosine similarity between embeddings)
    for i, item in faq_vector_store.items():
        similarity = np.dot(user_embedding, item["embedding"]) / (np.linalg.norm(user_embedding) * np.linalg.norm(item["embedding"]))
        similarities.append((similarity, item["answer"]))

    # Sort by similarity and return the most similar answer
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[0][1]  # Return the most similar answer

# --- Create Conversational Retrieval Chain ---
def qa_chain(user_input):
    # Retrieve the most relevant answer for the user's question
    answer = retrieve_answer(user_input)
    
    # Add the user input and answer to the conversation memory
    memory.add_user_message(user_input)
    memory.add_ai_message(answer)
    
    return answer

# --- Streamlit Frontend ---
st.title("AI Customer Support Chatbot")

# User Input for chatbot
user_input = st.text_input("Ask a question:")

if user_input:
    # Get the answer from the conversational retrieval chain
    response = qa_chain(user_input)
    st.write(response)

# Display previous messages from the conversation history
if memory.buffer:
    st.subheader("Conversation History:")
    for msg in memory.buffer:
        st.write(f"{msg['role']}: {msg['content']}")

