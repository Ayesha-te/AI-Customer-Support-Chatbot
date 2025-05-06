import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# Streamlit page setup
st.set_page_config(page_title="Universal AI Assistant 🌐", page_icon="🤖")
st.title("🤖 Universal AI Assistant")
st.sidebar.header("🛠 Controls")

# API key from .streamlit/secrets.toml
openai_api_key = st.secrets["openai_api_key"]

# Initialize LLM and memory
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Clear chat
if st.sidebar.button("🧹 Clear Chat"):
    memory.clear()

# User input
user_input = st.text_input("💬 Ask me anything:")

# Main logic
if user_input:
    # Append user message and call model
    messages = memory.chat_memory.messages
    messages.append(HumanMessage(content=user_input))

    response = llm(messages)

    # Save model's response to memory
    messages.append(AIMessage(content=response.content))

    # Show response
    st.markdown(f"**🤖 Assistant:** {response.content}")

# Show chat history
if memory.buffer:
    st.subheader("📜 Chat History")
    for msg in memory.buffer:
        if msg.type == "human":
            st.markdown(f"**🧑 You:** {msg.content}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg.content}")
