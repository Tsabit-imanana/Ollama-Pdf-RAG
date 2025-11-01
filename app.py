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

after_rag_template = """Anda adalah asisten profesional. Gunakan bahasa Indonesia formal namun bersahabat. Jawab hanya berdasarkan KONTEKS yang diberikan — jangan berspekulasi atau menambahkan informasi di luar konteks. Jangan menyebutkan nama dokumen atau sumber.

Instruksi output (WAJIB): Keluarkan satu objek JSON tunggal saja (tanpa teks tambahan). Format harus persis:
{{
  "summary": "<ringkasan 1-2 kalimat atau pesan-not-found>",
  "details": ["<poin pendukung 1>", "<poin pendukung 2>", ...],
  "evidence": ["<cuplikan konteks 1 (maks 200 karakter)>", ...],
  "conflicts": ["<perbedaan 1>", ...],
  "recommendation": "<rekomendasi singkat atau empty string>"
}}

Jika konteks tidak memadai, keluarkan persis:
{{
  "summary": "Maaf, saya tidak menemukan informasi terkait dalam dokumen. Silakan hubungi CS Trustmedis untuk bantuan lebih lanjut.",
  "details": [],
  "evidence": [],
  "conflicts": [],
  "recommendation": ""
}}

Aturan:
- Jawab hanya dari KONTEKS. Tidak ada spekulasi.
- Jangan menyebutkan nama dokumen atau sumber.
- "details", "evidence", dan "conflicts" harus array string.
- Evidence adalah potongan teks singkat (maks 200 karakter) yang mendukung poin; tidak menyertakan metadata file.
- Maksimal total item dalam details+conflicts+evidence: 6–8 elemen gabungan.
- Summary maksimal 1–2 kalimat.

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
    allow_origins=["*"],  # Ganti "*" dengan domain Laravel jika perlu
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode HTTP
    allow_headers=["*"],  # Izinkan semua header
)

# In-memory chat history store (for demo)
chat_histories = {}

def format_chat_history(history):
    # Format as string for prompt; gunakan ringkasan tiap jawaban jika tersedia
    lines = []
    for q, a in history:
        if isinstance(a, dict):
            summary = a.get("summary", "")
        else:
            summary = str(a)
        lines.append(f"User: {q}\nBot: {summary}")
    return "\n".join(lines)

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

        # invoke chain (model diinstruksikan mengeluarkan JSON)
        answer_raw = after_rag_chain.invoke({"question": question, "chat_history": chat_history_str})

        # Parse JSON output; jika gagal, fallback ke struktur aman
        parsed_answer = None
        if isinstance(answer_raw, dict):
            parsed_answer = answer_raw
        else:
            # try direct json
            try:
                parsed_answer = json.loads(answer_raw)
            except Exception:
                # try to recover JSON substring
                try:
                    start = answer_raw.find("{")
                    end = answer_raw.rfind("}") + 1
                    if start != -1 and end != -1:
                        parsed_answer = json.loads(answer_raw[start:end])
                except Exception:
                    parsed_answer = None

        if not isinstance(parsed_answer, dict):
            # fallback: wrap raw text into summary
            parsed_answer = {
                "summary": answer_raw if isinstance(answer_raw, str) else str(answer_raw),
                "details": [],
                "evidence": [],
                "conflicts": [],
                "recommendation": ""
            }

        # Ensure required fields exist and types are correct
        parsed_answer.setdefault("summary", "")
        parsed_answer.setdefault("details", [])
        parsed_answer.setdefault("evidence", [])
        parsed_answer.setdefault("conflicts", [])
        parsed_answer.setdefault("recommendation", "")

        # store structured answer in history (so next prompts get summaries)
        history.append((question, parsed_answer))
        chat_histories[session_id] = history

        # return structured response for frontend
        return {
            "answer": parsed_answer,
            "chat_history": [
                {"index": i + 1, "question": q, "answer": a}
                for i, (q, a) in enumerate(history)
            ],
        }
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