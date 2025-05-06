import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import yfinance as yf
import requests

# --- Load API Keys ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]

# --- Pinecone setup ---
pc = PineconeClient(api_key=PINECONE_API_KEY)

index_name = "customer-support-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# --- Embeddings and VectorStore ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# --- LLM and Memory ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=True
)

# --- Helper Functions ---
def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.info.get("regularMarketPrice")
        return f"The current price of {symbol.upper()} is ${price}" if price else "Stock symbol not found."
    except Exception:
        return "Error retrieving stock data."

def get_weather(city):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        response = requests.get(url)
        data = response.json()
        if "current" in data:
            temp_c = data["current"]["temp_c"]
            condition = data["current"]["condition"]["text"]
            return f"The current weather in {city} is {temp_c}°C with {condition}."
        else:
            return "City not found or weather API limit reached."
    except Exception:
        return "Error retrieving weather data."

# --- Streamlit UI ---
st.set_page_config(page_title="AI Support Chatbot", page_icon="🤖")
st.title("🤖 AI Customer Support Chatbot")
st.markdown("Ask about products, stock prices, weather, or general FAQs.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", placeholder="Ask a question...")

if user_input:
    # Check for weather or stock queries
    if "weather" in user_input.lower():
        city = user_input.split("in")[-1].strip()
        response = get_weather(city)
    elif any(term in user_input.lower() for term in ["stock", "price of", "share"]):
        words = user_input.upper().split()
        symbol = next((word for word in words if word.isalpha() and len(word) <= 5), "AAPL")
        response = get_stock_price(symbol)
    else:
        result = qa_chain({"question": user_input})
        response = result["answer"]

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# --- Display chat history ---
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")


