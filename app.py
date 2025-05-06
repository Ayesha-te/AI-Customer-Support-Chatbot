import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import yfinance as yf
import requests

# --- Load API Keys ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]

# --- Initialize Pinecone Client (v3) ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "customer-support-chatbot"

# Create index if not exists (one-time setup)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )

# Connect to the index
index = pc.Index(index_name)

# --- LangChain Embeddings & VectorStore ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Use PineconeVectorStore for existing index
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# --- Setup Memory and LLM ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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

user_input = st.text_input("Ask your question (FAQ, weather, stocks, general queries):", key="input")

# --- Real-time Data Handlers ---
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
    response = requests.get(url)
    data = response.json()
    if "current" in data:
        temp_c = data['current']['temp_c']
        condition = data['current']['condition']['text']
        return f"🌤️ The current temperature in {city} is {temp_c}°C with {condition}."
    else:
        return "❌ Could not fetch weather data."

def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    if not data.empty:
        price = data['Close'].iloc[-1]
        return f"📈 Current stock price of {ticker.upper()} is ${price:.2f}"
    else:
        return "❌ Could not fetch stock data."

# --- Process User Input ---
def process_query(query):
    query_lower = query.lower()
    
    if "weather" in query_lower:
        for word in query_lower.split():
            if word[0].isalpha() and word != "weather":
                return get_weather(word)
        return "🌍 Please mention the city for weather information."
    
    elif "stock" in query_lower or "price" in query_lower:
        for word in query.split():
            if word.isalpha() and len(word) <= 5:
                return get_stock_price(word)
        return "📊 Please mention the stock symbol (e.g., AAPL, TSLA)."
    
    else:
        return qa_chain.run(query)

# --- Handle Chat ---
if user_input:
    response = process_query(user_input)
    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("Bot", response))

# --- Display Chat ---
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}**: {message}")
