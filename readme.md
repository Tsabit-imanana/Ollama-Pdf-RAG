# PDF Chatbot RAG Project

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of PDF files using FastAPI (backend) and Streamlit (frontend). It uses LangChain, Ollama, and ChromaDB.

## Features

- Upload PDF files to the `pdfs` folder
- Ask questions via API or Streamlit UI
- Answers are generated based only on the PDF content
- Supports Bahasa Indonesia

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and set up Ollama

Follow [Ollama installation instructions](https://github.com/ollama/ollama/blob/main/docs/linux.md) for your OS.

#### Pull the chat model (e.g., gemma3):

```bash
ollama pull gemma3
```

#### Pull the embedding model (e.g., nomic-embed-text):

```bash
ollama pull nomic-embed-text
```

### 5. Add your PDF files

Place all your PDF files in the `pdfs` folder.

## Usage

### 1. Start the FastAPI backend

```bash
uvicorn app:app --reload
```

### 2. Start the Streamlit frontend (in a new terminal)

```bash
streamlit run ui.py
```

### 3. Ask questions

- Use the Streamlit UI in your browser (`http://localhost:8501`)
- Or send POST requests to the API endpoint (`http://localhost:8000/ask`)

**Example POST request:**
```json
{
  "question": "Apa isi dokumen ini?"
}
```

## Notes

- Add new PDFs to the `pdfs` folder and restart the backend to include them.
- Answers are based only on the content of the PDFs.