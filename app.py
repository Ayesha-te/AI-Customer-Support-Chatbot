import openai
import streamlit as st
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# --- Config ---
st.set_page_config(page_title="AI Customer Support 🤖", page_icon="💬")
st.title("🤖 AI Customer Support Chatbot")

# --- Secrets ---
openai_api_key = st.secrets["openai_api_key"]

# --- Models ---
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key)

# --- FAQ Data ---
faq_documents = [
    {"question": "What are your business hours?", "answer": "We are open from 9 AM to 5 PM, Monday to Friday. ⏰"},
    {"question": "What is your return policy?", "answer": "Our return policy lasts 30 days from the purchase date. 🔄"},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship internationally, with extra fees depending on the country. 🌍✈️"},
    {"question": "How can I contact customer support?", "answer": "You can reach us by email at support@example.com. 📧"}
]

faq_questions = [doc["question"] for doc in faq_documents]
faq_embeddings = [embeddings.embed_query(q) for q in faq_questions]

faq_vector_store = {
    i: {"question": faq_questions[i], "embedding": faq_embeddings[i], "answer": faq_documents[i]["answer"]}
    for i in range(len(faq_documents))
}

# --- Memory ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Clear button ---
st.sidebar.header("🛠 Controls")
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.memory.clear()
    st.session_state.user_input = ""
    st.experimental_rerun()

# --- Similarity Matching ---
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm else 0.0

def retrieve_answer(user_question):
    user_embedding = embeddings.embed_query(user_question)
    similarities = []

    for item in faq_vector_store.values():
        sim = cosine_similarity(user_embedding, item["embedding"])
        similarities.append((sim, item["answer"]))

    similarities.sort(reverse=True)
    top_sim, top_ans = similarities[0]
    return top_ans if top_sim > 0.75 else "Sorry, I couldn't find a clear answer to that. 🤔"

# --- Chat logic ---
def qa_chain(user_input):
    answer = retrieve_answer(user_input)
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(answer)
    return answer

# --- Input box ---
user_input = st.text_input("💬 Ask a question:", key="user_input")

# --- Output ---
if user_input:
    response = qa_chain(user_input)
    st.markdown(f"**🤖 Assistant:** {response}")

# --- History ---
if st.session_state.memory.buffer:
    st.subheader("📜 Chat History")
    for msg in st.session_state.memory.buffer:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**🗣 You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**🤖 Assistant:** {msg.content}")

