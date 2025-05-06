import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import pinecone
import requests
import os

# Set up API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Load vector store
index_name = "customer-support-chatbot"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if index_name not in pinecone.list_indexes():
    # Example: index creation if it doesn't exist (manual step recommended)
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# Setup memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup Chat Model
llm = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Customer Support Chatbot", page_icon="🤖")
st.title("🤖 AI Customer Support Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me anything (FAQ, weather, stock market):", key="input")

def fetch_realtime_data(query):
    if "weather" in query.lower():
        return "☁️ Real-time weather info (demo)"
    elif "stock" in query.lower():
        return "📈 Real-time stock data (demo)"
    return None

if user_input:
    # Check for real-time intent
    realtime_response = fetch_realtime_data(user_input)
    if realtime_response:
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Bot", realtime_response))
    else:
        response = qa_chain.run(user_input)
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Bot", response))

# Chat display
for sender, message in st.session_state.chat_history:
    st.write(f"**{sender}**: {message}")
