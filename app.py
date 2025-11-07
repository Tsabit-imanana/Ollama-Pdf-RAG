from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import os
import json


model_local = ChatOllama(model="deepseek-r1")

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

# Updated prompt: model MUST output a single JSON object {"answer":"<jawaban text>"} only.
after_rag_template = """Anda adalah asisten profesional. Gunakan bahasa Indonesia formal namun bersahabat. Jawab hanya berdasarkan KONTEKS yang diberikan — jangan berspekulasi atau menambahkan informasi di luar konteks. Jangan menyebutkan nama dokumen atau sumber.

Instruksi output (WAJIB): Keluarkan satu objek JSON tunggal saja (tanpa teks tambahan). Format harus persis:
{{ 
  "answer": "<jawaban singkat berbentuk teks>"
}}

Jika konteks tidak memadai, keluarkan persis:
{{ 
  "answer": "Maaf, saya tidak menemukan informasi terkait dalam dokumen. Silakan hubungi CS Trustmedis untuk bantuan lebih lanjut."
}}

Aturan:
- Jawab hanya dari KONTEKS. Tidak ada spekulasi.
- Jawaban adalah teks bebas jelaskan sedetail mungkin dalam Bahasa Indonesia formal.
- Buat jawaban seperti anda sendang melakukan tutorial step by step
- tutorial step by step harus diurutkan dengan nomor
- Jika ada daftar, gunakan bullet points.
- Jika konteks tidak memadai, keluarkan jawaban default di atas.
- Jangan keluarkan field selain 'answer'.
- Riwayat chat (ringkasan jawaban sebelumnya) akan disisipkan di bawah.
  
Riwayat chat:
{chat_history}

Konteks:
{context}

Pertanyaan:
{question}
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tetap "*"
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode HTTP
    allow_headers=["*"],  # Izinkan semua header
)

# In-memory chat history store (for demo)
chat_histories = {}

def format_chat_history(history):
    # Format as string for prompt; gunakan jawaban singkat tiap entry
    lines = []
    for q, a in history:
        lines.append(f"User: {q}\nBot: {a}")
    return "\n".join(lines)

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        session_id = data.get("session_id", "default")

        if not question:
            default_msg = "Maaf, saya tidak menemukan informasi terkait dalam dokumen. Silakan hubungi CS Trustmedis untuk bantuan lebih lanjut."
            return JSONResponse({"answer": default_msg, "chat_history": []}, status_code=400)

        history = chat_histories.get(session_id, [])
        chat_history_str = format_chat_history(history)

        # invoke chain (model diinstruksikan mengeluarkan JSON {"answer":"..."})
        answer_raw = after_rag_chain.invoke({"question": question, "chat_history": chat_history_str})

        # Parse JSON output to extract "answer" string; robust fallback to default message
        extracted_answer = None
        if isinstance(answer_raw, dict):
            extracted_answer = answer_raw.get("answer", None)
        else:
            # try direct json parse
            try:
                parsed = json.loads(answer_raw)
                if isinstance(parsed, dict) and "answer" in parsed:
                    extracted_answer = parsed.get("answer")
            except Exception:
                # try to recover JSON substring
                try:
                    start = answer_raw.find("{")
                    end = answer_raw.rfind("}") + 1
                    if start != -1 and end != -1:
                        parsed = json.loads(answer_raw[start:end])
                        if isinstance(parsed, dict) and "answer" in parsed:
                            extracted_answer = parsed.get("answer")
                except Exception:
                    extracted_answer = None

        # Fallbacks
        if not extracted_answer or not isinstance(extracted_answer, str):
            # If model returned something but not parsable, use raw string if available
            if isinstance(answer_raw, str) and answer_raw.strip():
                extracted_answer = answer_raw.strip()
            else:
                extracted_answer = "Maaf, saya tidak menemukan informasi terkait dalam dokumen. Silakan hubungi CS Trustmedis untuk bantuan lebih lanjut."

        # store answer string in history
        history.append((question, extracted_answer))
        chat_histories[session_id] = history

        formatted_history = [
            {"index": i + 1, "question": q, "answer": a}
            for i, (q, a) in enumerate(history)
        ]

        # Return only answer (string) and chat_history
        return {"answer": extracted_answer, "chat_history": formatted_history}
    except Exception as e:
        print("Error in /ask:", e)
        default_msg = "Maaf, saya tidak menemukan informasi terkait dalam dokumen. Silakan hubungi CS Trustmedis untuk bantuan lebih lanjut."
        current_history = chat_histories.get(data.get("session_id", "default")) if 'data' in locals() else []
        formatted_history = [
            {"index": i + 1, "question": q, "answer": a}
            for i, (q, a) in enumerate(current_history)
        ]
        return JSONResponse({"answer": default_msg, "chat_history": formatted_history}, status_code=500)

@app.get("/test")
async def test_connection():
    return {"answer": "ok", "chat_history": []}

@app.post("/history")
async def get_history(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id", "default")
        history = chat_histories.get(session_id, [])
        # Format with index; answer is string
        formatted_history = [
            {"index": i + 1, "question": q, "answer": a}
            for i, (q, a) in enumerate(history)
        ]
        return {"chat_history": formatted_history}
    except Exception as e:
        print("Error in /history:", e)
        return JSONResponse({"error": str(e)}, status_code=500)