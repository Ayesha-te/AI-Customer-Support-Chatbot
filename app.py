import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
import yfinance as yf
import requests

# --- Secrets ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]

# --- Vector Store Setup ---
index_name = "customer-support-chatbot"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Let LangChain handle Pinecone internally (don’t use pc = Pinecone(...))
vectorstore = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# --- LLM & Memory ---
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# --- Stock & Weather Functions ---
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

# --- Streamlit Chat UI ---
st.title("🤖 AI Customer Support Chatbot")
st.markdown("Ask anything about products, weather, or stock prices.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", placeholder="Ask me something...")

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

