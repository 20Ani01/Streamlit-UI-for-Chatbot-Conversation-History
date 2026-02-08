import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uuid
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from dotenv import load_dotenv
load_dotenv()

# ------------------ LLM ------------------
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=groq_api_key
)

# ------------------ UI ------------------
st.title("ChatBot Assistant")

# ------------------ Session Store ------------------
if "store" not in st.session_state:
    st.session_state.store = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if st.button("New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.store[st.session_state.session_id] = ChatMessageHistory()
    st.rerun()

session_id = st.session_state.session_id

# ------------------ Memory ------------------
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ------------------ Chat Form (IMPORTANT PART) ------------------
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your Query.........")
    submit = st.form_submit_button("Send")

# ------------------ LLM Call ------------------
if submit and user_input:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. First answer the query in {lang1}, then translate it in {lang2}"
        ),
        ("user", "{context}")
    ])

    chain = prompt | llm

    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="context"
    )

    response = conversation.invoke(
        {
            "lang1": "English",
            "lang2": "Spanish",
            "context": user_input,
        },
        config={
            "configurable": {"session_id": session_id}
        }
    )

    st.write(response.content)
