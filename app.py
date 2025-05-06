import openai
import streamlit as st
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Setup Streamlit
st.set_page_config(page_title="AI Customer Support 🤖", page_icon="💬")
st.title("🤖 AI Customer Support Chatbot")
st.sidebar.header("🛠 Controls")

# Get API Key from secrets.toml
openai_api_key = st.secrets["openai_api_key"]

# Embedding + LLM
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Dummy FAQ Data
faq_documents = [
    {"question": "What are your business hours?", "answer": "We are open from 9 AM to 5 PM, Monday to Friday. ⏰"},
    {"question": "What is your return policy?", "answer": "Our return policy lasts 30 days from the purchase date. 🔄"},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship internationally, with extra fees depending on the country. 🌍✈️"},
    {"question": "How can I contact customer support?", "answer": "You can reach us by email at support@example.com. 📧"},
]

faq_questions = [doc["question"] for doc in faq_documents]

# Cache Embeddings
@st.cache_data
def get_faq_embeddings():
    return [embeddings.embed_query(q) for q in faq_questions]

faq_embeddings = get_faq_embeddings()

# In-memory vector store
faq_vector_store = {
    i: {"question": faq_questions[i], "embedding": faq_embeddings[i], "answer": faq_documents[i]["answer"]}
    for i in range(len(faq_documents))
}

# Memory stored in session
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Similarity Function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Retrieve Best Match
def retrieve_answer(user_question):
    user_embedding = embeddings.embed_query(user_question)
    similarities = []

    for item in faq_vector_store.values():
        sim = cosine_similarity(user_embedding, item["embedding"])
        similarities.append((sim, item["answer"]))
        print(f"Similarity with '{item['question']}': {sim:.4f}")

    similarities.sort(reverse=True)
    top_sim, top_ans = similarities[0]
    print(f"Top similarity: {top_sim:.4f}")
    return top_ans if top_sim > 0.75 else "Sorry, I couldn't find a clear answer to that. 🤔"

# Chat logic
def qa_chain(user_input):
    answer = retrieve_answer(user_input)
    st.session_state.memory.chat_memory.add_user_message(user_input)
    st.session_state.memory.chat_memory.add_ai_message(answer)
    return answer

# Clear Chat
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.memory.clear()
    st.experimental_rerun()

# User Input
user_input = st.text_input("💬 Ask a question:")

# Show response
if user_input:
    response = qa_chain(user_input)
    st.markdown(f"**🤖 Assistant:** {response}")

# Chat History
if st.session_state.memory.buffer:
    st.subheader("📜 Chat History")
    for msg in st.session_state.memory.buffer:
        if msg.type == "human":
            st.markdown(f"**🗣 You:** {msg.content}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg.content}")


