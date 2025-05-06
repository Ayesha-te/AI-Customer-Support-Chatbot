import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
import yfinance as yf
import requests
import pinecone

# --- Secrets from Streamlit ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]  # e.g., "us-east-1-aws"

# --- Initialize Pinecone SDK ---
from pinecone import Pinecone

# Initialize Pinecone using the correct method (Pinecone class)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# --- Initialize Pinecone Index ---
index_name = "customer-support-chatbot"

# Explicitly create the pinecone.Index instance
index = pinecone_client.Index(index_name)

# --- LangChain Embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- Vector Store using LangChain Community ---
vectorstore = PineconeVectorStore(
    index=index,  # Pass the correct pinecone.Index instance
    embedding=embeddings,
    text_key="text"  # Ensure this matches the field where your text data is stored
)

# --- Setup Memory and LLM ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# --- Stock and Weather Functions ---
def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.info.get("regularMarketPrice")
        return f"The current price of {symbol.upper()} is ${price}" if price else "Stock symbol not found."
    except:
        return "Error fetching stock price."

def get_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        res = requests.get(url).json()
        if "current" in res:
            return f"{city}: {res['current']['temp_c']}°C, {res['current']['condition']['text']}."
        return "City not found."
    except:
        return "Weather service error."

# --- Streamlit UI ---
st.title("🤖 AI Customer Support Chatbot")
st.markdown("Ask about your queries, products, stock prices, or weather.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", placeholder="Ask a question...")

if user_input:
    if "weather" in user_input.lower():
        city = user_input.split("in")[-1].strip()
        answer = get_weather(city)
    elif any(word in user_input.lower() for word in ["stock", "price", "share"]):
        symbol = user_input.upper().split()[-1]
        answer = get_stock_price(symbol)
    else:
        result = qa_chain({"question": user_input})
        answer = result["answer"]

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")

