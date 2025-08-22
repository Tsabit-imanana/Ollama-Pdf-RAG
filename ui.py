import streamlit as st
import requests

st.title("PDF Chatbot (Bahasa Indonesia)")

question = st.text_input("Masukkan pertanyaan:")

if st.button("Tanya"):
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question}
    )
    st.write(response.json().get("answer", "Tidak ada jawaban."))