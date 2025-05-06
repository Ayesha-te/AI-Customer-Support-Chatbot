import openai
import streamlit as st
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Get API Key from .streamlit/secrets.toml
openai_api_key = st.secrets["openai_api_key"]

# Initialize embedding and chat model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Dummy FAQ data
faq_documents = [
    {"question": "What are your business hours?", "answer": "We are open from 9 AM to 5 PM, Monday to Friday. ⏰"},
    {"question": "What is your return policy?", "answer": "Our return policy lasts 30 days from the purchase date. 🔄"},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship internationally, with extra fees depending on the country. 🌍✈️"},
    {"question": "How can I contact customer support?", "answer": "You can reach us by email at support@example.com. 📧"}
]

# Embed FAQ questions
faq_questions = [doc["question"] for doc in faq_documents]
faq_embeddings = [embeddings.embed_query(q) for q in faq_questions]

# In-memory store
faq_vector_store = {
    i: {"question": faq_questions[i], "embedding": faq_embeddings[i], "answer": faq_documents[i]["answer"]}
    for i in range(len(faq_documents))
}

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Find best match answer
def retrieve_answer(user_question):
    user_embedding = embeddings.embed_query(user_question)
    similarities = []

    for i, item in faq_vector_store.items():
        sim = np.dot(user_embedding, item["embedding"]) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(item["embedding"])
        )
        similarities.append((sim, item["answer"]))

    similarities.sort(reverse=True)
    return similarities[0][1] if similarities else "Sorry, I don't know the answer to that yet. 🤔"

# Main QA logic
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

user_input = st.text_input("💬 Ask a question:")

if user_input:
    answer = qa_chain(user_input)
    st.markdown(f"**🤖 Assistant:** {answer}")

if memory.buffer:
    st.subheader("📜 Chat History")
    for msg in memory.buffer:
        if msg.type == "human":
            st.markdown(f"**🗣 You:** {msg.content}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg.content}")


