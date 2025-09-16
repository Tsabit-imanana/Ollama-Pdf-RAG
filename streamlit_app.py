import streamlit as st
import requests
import uuid

st.title("RAG Chatbot")

# Generate or get session_id for the user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Fetch chat history from backend
def fetch_history():
    response = requests.post(
        "http://localhost:8000/history",
        json={"session_id": st.session_state.session_id}
    )
    if response.status_code == 200:
        return response.json().get("chat_history", [])
    return []

# Ask question to backend
def ask_question(question):
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question, "session_id": st.session_state.session_id}
    )
    if response.status_code == 200:
        return response.json().get("answer", "")
    return "Error: Tidak dapat mengambil jawaban."

# Display chat history
chat_history = fetch_history()
for chat in chat_history:
    st.markdown(f"**{chat['index']}. You:** {chat['question']}")
    st.markdown(f"**{chat['index']}. Bot:** {chat['answer']}")

# User input
question = st.text_input("Tanyakan sesuatu...", key="input")

if st.button("Kirim"):
    if question:
        answer = ask_question(question)
        # No need to update local history, just rerun to fetch from backend
        st.experimental_rerun()

# Generate follow-up question (optional)
if chat_history:
    last_answer = chat_history[-1]["answer"]
    followup = f"Apa pertanyaan lanjutan dari jawaban: '{last_answer}'?"
    if st.button("Pertanyaan lanjutan"):
        answer = ask_question(followup)
        st.experimental_rerun()