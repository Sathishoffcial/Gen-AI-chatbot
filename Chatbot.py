
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="GEN AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 GEN AI CHATBOT")
st.caption("Powered by Groq + LangChain")

# Check API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    api_key=GROQ_API_KEY
)

# Sidebar
with st.sidebar:
    st.header("Chat Settings")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.write("Model: GPT-OSS 120B")
    st.write("Provider: Groq")

# Display previous messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_prompt = st.chat_input("Ask me anything...")

if user_prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Save user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_prompt
    })

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            try:
                response = llm.invoke(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful AI assistant. "
                                "Give clear, accurate and easy-to-understand answers."
                            )
                        },
                        *st.session_state.chat_history
                    ]
                )

                assistant_response = response.content

                st.markdown(assistant_response)

                # Save assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })

            except Exception as e:
                st.error(f"Error generating response: {e}")
