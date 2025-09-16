from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os

model_local = ChatOllama(model="mistral")

pdf_folder = "pdfs"
pdf_files = [
    os.path.join(pdf_folder, f)
    for f in os.listdir(pdf_folder)
    if f.lower().endswith(".pdf")
]

docs = [PyPDFLoader(pdf).load() for pdf in pdf_files]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=5500, chunk_overlap=250)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
)
retriever = vectorstore.as_retriever()

after_rag_template = """Jawab pertanyaan berikut secara langsung, singkat, dan jelas dalam bahasa Indonesia, hanya berdasarkan konteks yang diberikan. Jangan sebutkan sumber atau dokumen.

Riwayat chat:
{chat_history}
Konteks:
{context}
Pertanyaan: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)

app = FastAPI()

# In-memory chat history store (for demo)
chat_histories = {}

def format_chat_history(history):
    # Format as string for prompt
    return "\n".join([f"User: {q}\nBot: {a}" for q, a in history])

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        session_id = data.get("session_id", "default")

        if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)

        history = chat_histories.get(session_id, [])
        chat_history_str = format_chat_history(history)

        answer = after_rag_chain.invoke({"question": question, "chat_history": chat_history_str})

        history.append((question, answer))
        chat_histories[session_id] = history

        return {"answer": answer, "chat_history": history}
    except Exception as e:
        print("Error in /ask:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/test")
async def test_connection():
    return {"status": "ok", "message": "Connection successful"}

@app.post("/history")
async def get_history(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id", "default")
        history = chat_histories.get(session_id, [])
        # Format with index
        formatted_history = [
            {"index": i + 1, "question": q, "answer": a}
            for i, (q, a) in enumerate(history)
        ]
        return {"chat_history": formatted_history}
    except Exception as e:
        print("Error in /history:", e)
        return JSONResponse({"error": str(e)}, status_code=500)