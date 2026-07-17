import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

# Initialize model
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.9
)

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Chatbot")

# ---------------- AI Mode ---------------- #

mode = st.selectbox(
    "Choose your AI mode",
    (
        "Helpful Chatbot",
        "Funny Chatbot",
        "Angry Chatbot",
        "Sad Chatbot",
    ),
)

system_prompts = {
    "Helpful Chatbot": "You are a helpful AI agent.",
    "Funny Chatbot": "You are a funny AI agent.",
    "Angry Chatbot": "You are an angry AI agent.",
    "Sad Chatbot": "You are a sad AI agent.",
}

# Initialize chat history
if (
    "messages" not in st.session_state
    or st.session_state.get("current_mode") != mode
):
    st.session_state.current_mode = mode
    st.session_state.messages = [
        SystemMessage(content=system_prompts[mode])
    ]

# ---------------- Display Chat ---------------- #

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# ---------------- Chat Input ---------------- #

user_prompt = st.chat_input("Type your message...")

if user_prompt:

    with st.chat_message("user"):
        st.markdown(user_prompt)

    st.session_state.messages.append(
        HumanMessage(content=user_prompt)
    )

    response = model.invoke(st.session_state.messages)

    st.session_state.messages.append(
        AIMessage(content=response.content)
    )

    with st.chat_message("assistant"):
        st.markdown(response.content)
