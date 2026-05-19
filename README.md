# RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** web app that lets you upload a PDF and ask questions about its content. Answers are grounded in retrieved document chunks, powered by **Groq (Llama 3.3 70B)** for generation and **FAISS + Sentence Transformers** for semantic search.

## Features

- **PDF upload** via Streamlit sidebar
- **Semantic search** with FAISS (`all-MiniLM-L6-v2` embeddings)
- **MMR retrieval** — diverse, relevant chunks with a small `k` (token-efficient)
- **Extractive context compression** — ranks sentences locally before sending text to Groq (no extra API call)
- **Cached index** — re-indexes only when you upload a new or changed PDF
- **List-aware answers** — numbered lists for enumeration-style questions; normal prose for others
- **Transparency** — expander shows compressed context and raw retrieved chunks

## Tech stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| LLM | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) (CPU) |
| Orchestration | [LangChain](https://www.langchain.com/) (loaders, splitters, FAISS wrapper) |
| PDF parsing | [PyPDF](https://pypi.org/project/pypdf/) |

## Project structure

```
rag2/
├── README.md
├── LICENSE
├── requirements.txt
└── rag_app/
    ├── app.py          # Streamlit RAG application
    ├── .env            # Groq API key (not committed)
    └── .gitignore
```

## Prerequisites

- Python 3.10+ recommended
- A [Groq API key](https://console.groq.com/)

## Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd rag2
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create `rag_app/.env`:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Run the app

```bash
cd rag_app
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

1. Upload a PDF in the sidebar.
2. Wait for indexing to finish.
3. Ask questions in the chat input.

Use **Clear Chat** to reset conversation history (the PDF index stays cached).

## How it works

```
PDF → chunk → embed → FAISS index
                              ↓
User question → MMR retrieval → extractive compression → Groq → answer
```

1. **Ingestion** — PDF pages are split into overlapping text chunks and embedded.
2. **Retrieval** — The question is normalized (e.g. `what are the types of…` → `types of…`), then **Maximal Marginal Relevance (MMR)** selects a small set of diverse chunks.
3. **Compression** — The most query-relevant sentences are kept up to a character budget (local embeddings; no second Groq call).
4. **Generation** — One Groq request answers from the compressed context only.

## Configuration

Tune constants at the top of `rag_app/app.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CHUNK_SIZE` | 1000 | Characters per chunk (re-upload PDF after change) |
| `CHUNK_OVERLAP` | 300 | Overlap between chunks |
| `DEFAULT_K` | 5 | MMR chunks for normal questions |
| `LIST_QUERY_K` | 8 | MMR chunks for list-style questions |
| `MMR_FETCH_K` | 20 | Candidate pool for MMR |
| `MAX_CONTEXT_CHARS` | 5000 | Max text sent to Groq (normal) |
| `MAX_CONTEXT_CHARS_LIST` | 7000 | Max text sent to Groq (lists) |

The sidebar **Retrieval & limits** expander shows current values at runtime.



Answers are limited to retrieved context. If information is missing from the PDF, the model should say the document does not contain it.

## Troubleshooting

### Groq rate limits

Large chunks, high `k`, and huge context increase tokens per request. This app uses **small k**, **MMR**, **context caps**, and **one Groq call per question** to reduce usage. If you still hit limits:

- Lower `MAX_CONTEXT_CHARS` / `DEFAULT_K`
- Wait for your Groq quota to reset
- Check usage on the [Groq console](https://console.groq.com/)


