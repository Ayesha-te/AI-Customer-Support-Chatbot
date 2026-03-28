import streamlit as st
from openai import AuthenticationError, OpenAI


def get_openai_api_key() -> str:
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    if "openai_api_key" in st.secrets:
        return st.secrets["openai_api_key"]
    if "openai" in st.secrets:
        openai_section = st.secrets["openai"]
        if "apikey" in openai_section:
            return openai_section["apikey"]
        if "api_key" in openai_section:
            return openai_section["api_key"]

    st.error(
        "Missing OpenAI API key in Streamlit secrets. Add one of: "
        "`OPENAI_API_KEY`, `openai_api_key`, or `[openai] apikey`."
    )
    st.stop()


def get_client() -> OpenAI:
    return OpenAI(api_key=get_openai_api_key())


st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("🤖 Ask Me Anything - AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

user_input = st.text_input("💬 Ask your question:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = get_client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a helpful customer support assistant."},
                *st.session_state.messages,
            ],
        )
    except AuthenticationError:
        st.error(
            "OpenAI authentication failed. Check that your Streamlit secret contains a valid OpenAI API key."
        )
        st.stop()

    assistant_reply = response.choices[0].message.content or ""
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.markdown(f"**🤖 Assistant:** {assistant_reply}")

if st.session_state.messages:
    st.subheader("📜 Chat History")
    for msg in st.session_state.messages:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        st.markdown(f"**{role}:** {msg['content']}")
