import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

# Streamlit UI setup
st.set_page_config(page_title="AI Assistant 🌐", page_icon="🤖")
st.title("🤖 Ask Me Anything - AI Assistant")

# Get OpenAI API Key securely from secrets.toml
openai_api_key = st.secrets["openai_api_key"]

# Initialize LLM and memory
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
memory = ConversationBufferMemory(return_messages=True)

# Button to clear chat
if st.sidebar.button("🧹 Clear Chat"):
    memory.chat_memory.messages = []  # Properly reset messages list

# Get user input
user_input = st.text_input("💬 Ask your question:")

# If input is given
if user_input:
    # Add user's question to memory
    memory.chat_memory.messages.append(HumanMessage(content=user_input))

    # Get LLM response
    response = llm(memory.chat_memory.messages)

    # Save response in memory
    memory.chat_memory.messages.append(AIMessage(content=response.content))

    # Display assistant reply
    st.markdown(f"**🤖 Assistant:** {response.content}")

# Show chat history
if memory.chat_memory.messages:
    st.subheader("📜 Chat History")
    for msg in memory.chat_memory.messages:
        role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg.content}")
