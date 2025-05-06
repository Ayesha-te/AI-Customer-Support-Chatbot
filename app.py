from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# Keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]

# Init Pinecone v3
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "customer-support-chatbot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(index_name)

# Embedder
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Vectorstore (compatible with LangChain)
vectorstore = Pinecone(index, embeddings.embed_query, "text")
