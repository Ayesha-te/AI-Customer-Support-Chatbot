import openai
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import InMemoryVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings  # Update imports based on LangChain warnings

# Set up your OpenAI API key (make sure to keep it in the secrets file)
openai_api_key = st.secrets["openai_api_key"]

# --- Setup OpenAI Embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# --- Example FAQ Documents ---
faq_documents = [
    {"question": "What are your business hours?", "answer": "We are open from 9 AM to 5 PM, Monday to Friday."},
    {"question": "What is your return policy?", "answer": "Our return policy lasts 30 days from the purchase date."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship internationally, with extra fees depending on the country."},
    {"question": "How can I contact customer support?", "answer": "You can reach us by email at support@example.com."}
]

# Embedding FAQ questions for vector search
faq_questions = [doc['question'] for doc in faq_documents]
faq_embeddings = [embeddings.embed_query(q) for q in faq_questions]

# --- Create In-Memory Vector Store ---
vector_store = InMemoryVectorStore(embedding_function=embeddings.embed_query)

# Add FAQ documents to the vector store
for i, faq in enumerate(faq_documents):
    vector_store.add_texts([faq["question"]], [faq["answer"]])

# --- Setup Memory for Conversation ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Setup LangChain LLM ---
llm = ChatOpenAI(openai_api_key=openai_api_key)

# --- Create Conversational Retrieval Chain ---
qa_chain = ConversationalRetrievalChain.from_llm_and_vectorstore(
    llm=llm,
    vectorstore=vector_store,
    memory=memory
)

# --- Streamlit Frontend ---
st.title("AI Customer Support Chatbot")

# User Input for chatbot
user_input = st.text_input("Ask a question:")

if user_input:
    # Get the answer from the conversation chain
    response = qa_chain.run(input=user_input)
    st.write(response)

# Display previous messages from the conversation history
if memory.buffer:
    st.subheader("Conversation History:")
    for msg in memory.buffer:
        st.write(f"{msg['role']}: {msg['content']}")

