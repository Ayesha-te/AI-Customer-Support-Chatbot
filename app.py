import openai
import streamlit as st
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Get API Key from secrets.toml
openai_api_key = st.secrets["openai_api_key"]

# Initialize embeddings and chat model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Dummy FAQ data
faq_documents = [
    {"question": "What are your business hours?", "answer": "We are open from 9 AM to 5 PM, Monday to Friday. ⏰"},
    {"question": "What is your return policy?", "answer": "Our return policy lasts 30 days from the purchase date. 🔄"},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship internationally, with extra fees depending on the country. 🌍✈️"},
    {"question": "How can I contact customer support?", "answer": "You can reach us by email at support@example.com. 📧"}
]

# Pre-compute normalized FAQ embeddings
faq_vector_store = []
for doc in faq_documents:
    vec = embeddings.embed_query(doc["question"])
    norm_vec = vec / np.linalg.norm(vec)
    faq_vector_store.append({"question": doc["question"], "embedding": norm_vec, "answer": doc["answer"]})

# Memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Retrieve best matching answer with a confidence threshold
def retrieve_answer(user_question, threshold=0.75):
    user_vec = embeddings.embed_query(user_question)
    norm_user_vec = user_vec / np.linalg.norm(user_vec)

    best_sim = -1
    best_answer = "🤔 Sorry, I couldn't find a matching answer. Please try rephrasing."

    for item in faq_vector_store:
        sim = np.dot(norm_user_vec, item["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_answer = item["answer"] if sim >= threshold else best_answer

    return best_answer

# Handle user input and memory
def qa_chain(user_input):
    answer = retrieve_answer(user_input)
    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(answer)
    return answer

# --- Streamlit UI ---
st.set_page_config(page_title="AI Customer Support 🤖", page_icon="💬")
st.title("🤖 AI Customer Support Chatbot")

st.sidebar.header("🛠 Controls")
if st.sidebar.button("🧹 Clear Chat"):
    memory.clear()

st.markdown("Welcome! Ask me anything about our services. 🛍️")

user_input = st.text_input("💬 Ask a question:")

if user_input:
    answer = qa_chain(user_input)
    st.markdown(f"**🤖 Assistant:** {answer}")

# Display chat history
if memory.buffer:
    st.subheader("📜 Chat History")
    for msg in memory.buffer:
        if msg.type == "human":
            st.markdown(f"**🗣 You:** {msg.content}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg.content}")
