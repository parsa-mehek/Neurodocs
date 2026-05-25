# NeuraDocs

NeuraDocs is an AI-powered PDF assistant that allows users to upload documents, ask intelligent questions, generate practice questions, and simplify complex PDFs using semantic AI search.

Built with Python, Streamlit, FAISS, and Hugging Face Transformers.

---

## Features

- AI-powered PDF conversations
- Semantic document search
- Persistent chat history
- Multiple chat workspaces
- Upload multiple PDFs
- PDF preview support
- Generate questions from PDFs
- Simplify PDFs into easy language
- Rename and delete chats
- Modern SaaS-style UI
- Local vector search using FAISS
- Semantic embeddings with Sentence Transformers

---

## Tech Stack

| Technology | Usage |
|---|---|
| Python | Backend |
| Streamlit | UI Framework |
| PyPDF2 | PDF Text Extraction |
| Sentence Transformers | Text Embeddings |
| FAISS | Vector Similarity Search |
| Hugging Face Transformers | AI Response Generation |
| TinyDB | Persistent Local Database |
| CSS | Custom Modern UI |

---

## AI Architecture

This project uses a simplified RAG (Retrieval-Augmented Generation) pipeline.

### Workflow

```text
PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
FAISS Semantic Search
↓
Relevant Context Retrieval
↓
AI Response Generation
```

---

## Screenshots

Add screenshots here.

### Main Workspace

![Main UI](assets/main-ui.png)

### Chat Interface

![Chat UI](assets/chat-ui.png)

### PDF Preview

![PDF Preview](assets/pdf-preview.png)

---

## Installation

### Clone Repository

```bash
git clone YOUR_REPOSITORY_URL
cd neuradocs
```

---

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Project Structure

```text
neuradocs/
│
├── app.py
├── requirements.txt
├── README.md
├── chat_db.json
├── uploaded_pdfs/
├── .streamlit/
│   └── config.toml
└── assets/
```

---

## Features Explained

### AI PDF Chat

Users can upload PDFs and ask context-aware questions using semantic search and AI-generated responses.

---

### Question Generator

The application can generate:

- MCQs
- Short Questions
- Viva Questions

directly from PDF content.

---

### Easy Explanation Mode

Complex PDF content is rewritten into beginner-friendly explanations.

---

### Persistent Workspaces

Chats, PDFs, and histories are permanently stored locally using TinyDB.

---

## Models Used

### Embedding Model

```text
all-MiniLM-L6-v2
```

Used for semantic vector embeddings.

---

### Language Model

```text
google/flan-t5-small
```

Used for:
- answering questions
- generating explanations
- generating practice questions

---

## Future Improvements

- Streaming responses
- Chat search
- User authentication
- Cloud database
- OCR for scanned PDFs
- Markdown rendering
- Citation support
- Export chats
- Multi-user workspaces
- Voice input

---

## Why This Project Is Important

This project demonstrates practical AI engineering concepts including:

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- Embeddings
- Prompt Engineering
- Persistent AI Workspaces

---

## Author

Parsa Amin Mehek
