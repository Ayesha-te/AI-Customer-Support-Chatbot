import streamlit as st
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

# Page config
st.set_page_config(page_title="AI Customer Support 🤖", page_icon="💬")
st.title("🤖 AI Customer Support Chatbot")

# Load secrets
openai_api_key = st.secrets["openai_api_key"]

# Initialize model
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

# In-memory vector store
faq_vector_store = {
    i: {"question": faq_questions[i], "embedding": faq_embeddings[i], "answer": faq_documents[i]["answer"]}
    for i in range(len(faq_documents))
}

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = []

# Clear chat
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.memory = []

# Answer retrieval
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

# UI
user_input = st.text_input("💬 Ask a question:")

if user_input:
    answer = retrieve_answer(user_input)
    st.session_state.memory.append(("user", user_input))
    st.session_state.memory.append(("ai", answer))
    st.markdown(f"**🤖 Assistant:** {answer}")

if st.session_state.memory:
    st.subheader("📜 Chat History")
    for role, msg in st.session_state.memory:
        if role == "user":
            st.markdown(f"**🗣 You:** {msg}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg}")
