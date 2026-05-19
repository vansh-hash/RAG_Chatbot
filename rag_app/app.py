import os
import re
import tempfile

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Retrieval (re-upload PDF after changing chunk settings) ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

DEFAULT_K = 5
LIST_QUERY_K = 8
MMR_FETCH_K = 20          # pool size for MMR reranking
MMR_LAMBDA = 0.7          # 1.0 = max relevance, lower = more diversity

LIST_EXPAND_MAX_PAGES = 2   # mild page expansion for list questions only

# Cap text sent to Groq per request (extractive compression, no extra API call)
MAX_CONTEXT_CHARS = 5000
MAX_CONTEXT_CHARS_LIST = 7000

GROQ_MAX_TOKENS = 1024
GROQ_MAX_TOKENS_LIST = 2048

LIST_QUERY_PATTERNS = (
    "list all", "list the", "list ",
    "name all", "name the",
    "how many",
    "what are the", "what are all",
    "give me all", "tell me all",
    "types of", "categories of",
    "key terminolog",
    "steps to", "steps for", "steps in",
    "examples of", "advantages of", "disadvantages of",
    "applications of",
    "enumerate", "each of the", "every ",
)


def is_list_query(query: str) -> bool:
    q = query.lower()
    return any(pattern in q for pattern in LIST_QUERY_PATTERNS)


def normalize_query_for_retrieval(query: str) -> str:
    q = query.strip().rstrip("?").lower()
    prefixes = (
        "what are the ", "what are ", "what is the ", "what is ",
        "list the ", "list all ", "list ", "name the ", "name all ",
        "tell me the ", "tell me ", "give me the ", "give me ",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
                changed = True
                break
    return q.strip() or query.strip()


def dedupe_documents(documents):
    seen = set()
    unique = []
    for doc in documents:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def expand_same_page_chunks(retrieved_docs, all_docs, max_pages=2):
    if not retrieved_docs or not all_docs:
        return retrieved_docs

    pages = []
    seen = set()
    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        if page is None or page in seen:
            continue
        seen.add(page)
        pages.append(page)
        if len(pages) >= max_pages:
            break

    expanded = list(retrieved_docs)
    for doc in all_docs:
        if doc.metadata.get("page") in pages:
            expanded.append(doc)
    return dedupe_documents(expanded)


def sort_docs_by_page(docs):
    return sorted(
        docs,
        key=lambda d: (d.metadata.get("page", 0), d.page_content[:60]),
    )


def retrieve_documents_mmr(vectorstore, query: str, all_docs):
    """MMR: diverse, relevant chunks with a small k (fewer tokens to Groq)."""
    search_query = normalize_query_for_retrieval(query)
    is_list = is_list_query(query)
    k = LIST_QUERY_K if is_list else DEFAULT_K

    docs = vectorstore.max_marginal_relevance_search(
        search_query,
        k=k,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA,
    )
    docs = dedupe_documents(docs)

    if is_list:
        docs = sort_docs_by_page(
            expand_same_page_chunks(docs, all_docs, max_pages=LIST_EXPAND_MAX_PAGES)
        )

    return docs


def compress_context_extractive(query: str, docs, embedding_model, max_chars: int) -> str:
    """
    Keep the most query-relevant sentences up to max_chars.
    Runs locally (Sentence Transformers) — no extra Groq call.
    """
    if embedding_model is None:
        return trim_context_by_chars(docs, max_chars)

    sentences = []
    for doc in docs:
        for sent in re.split(r"(?<=[.!?])\s+", doc.page_content):
            sent = sent.strip()
            if len(sent) > 25:
                sentences.append(sent)

    if not sentences:
        return trim_context_by_chars(docs, max_chars)

    unique_sentences = list(dict.fromkeys(sentences))
    if len(" ".join(unique_sentences)) <= max_chars:
        return " ".join(unique_sentences)

    q_vec = np.array(embedding_model.embed_query(query))
    q_norm = np.linalg.norm(q_vec) + 1e-9

    scored = []
    batch_size = 32
    for i in range(0, len(unique_sentences), batch_size):
        batch = unique_sentences[i : i + batch_size]
        for sent, emb in zip(batch, embedding_model.embed_documents(batch)):
            s_vec = np.array(emb)
            sim = float(np.dot(q_vec, s_vec) / (q_norm * (np.linalg.norm(s_vec) + 1e-9)))
            scored.append((sim, sent))

    scored.sort(key=lambda x: x[0], reverse=True)

    parts, total = [], 0
    for _, sent in scored:
        if total + len(sent) + 1 > max_chars:
            continue
        parts.append(sent)
        total += len(sent) + 1

    return " ".join(parts) if parts else trim_context_by_chars(docs, max_chars)


def trim_context_by_chars(docs, max_chars: int) -> str:
    """Fallback: take whole chunks in order until the char budget is full."""
    parts, total = [], 0
    for doc in docs:
        text = doc.page_content.strip()
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text) + 2
    return "\n\n".join(parts)


class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


def groq_complete(prompt: str, temperature: float = 0.2, max_tokens: int = GROQ_MAX_TOKENS) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def build_prompt(query: str, context: str, as_list: bool = False) -> str:
    list_rules = ""
    if as_list:
        list_rules = """
- The user wants a LIST. Use a numbered markdown list (one item per line).
- Format: 1. **Term**: definition
- Include every relevant item from the context; read the full context first.
- Do not write one long paragraph.
"""

    return f"""
You are a document question-answering assistant.

IMPORTANT RULES:
- Answer ONLY from the provided context
- Do NOT use external knowledge
- If the answer is not in the context, say:
  "The document does not contain this information."
{list_rules}
Context:
{context}

Question:
{query}
"""


client = Groq(api_key=GROQ_API_KEY)


def get_embedding_model():
    """Load embeddings once; cached sessions from older builds may not have it."""
    if st.session_state.embedding_model is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embedding_model = SentenceTransformerEmbeddings(
                "all-MiniLM-L6-v2"
            )
    return st.session_state.embedding_model


# --- UI ---

st.sidebar.title("📄 RAG Chatbot")
st.sidebar.markdown(
    "Upload a PDF and chat with it using Groq, Llama 3.3 70B, FAISS, and Sentence Transformers."
)

with st.sidebar.expander("Retrieval & limits"):
    st.markdown(
        f"""
        **Indexing:** chunk {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}  
        **MMR:** k={DEFAULT_K} (normal) / {LIST_QUERY_K} (lists), fetch={MMR_FETCH_K}  
        **Context cap:** {MAX_CONTEXT_CHARS} / {MAX_CONTEXT_CHARS_LIST} chars  
        **Groq:** 1 call per question (extractive compression is local)  

        MMR gives diverse chunks with a small k → fewer tokens, less rate-limit pressure.  
        Re-upload PDF after changing chunk settings.
        """
    )

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

st.title("📚 Chat With Your PDF")
st.markdown("Ask questions from your uploaded document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "docs" not in st.session_state:
    st.session_state.docs = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "indexed_file_id" not in st.session_state:
    st.session_state.indexed_file_id = None

if uploaded_file is None:
    st.info("Please upload a PDF in the sidebar to begin.")
else:
    file_id = f"{uploaded_file.name}:{uploaded_file.size}"
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.session_state.indexed_file_id != file_id:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name

        with st.spinner("Loading PDF..."):
            documents = PyPDFLoader(pdf_path).load()
        st.info(f"Loaded {len(documents)} pages.")

        with st.spinner("Splitting document..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            docs = splitter.split_documents(documents)
        st.info(f"Created {len(docs)} chunks.")

        with st.spinner("Building index..."):
            embeddings = get_embedding_model()
            vectorstore = FAISS.from_documents(docs, embeddings)

        st.session_state.vectorstore = vectorstore
        st.session_state.docs = docs
        st.session_state.indexed_file_id = file_id
        st.session_state.messages = []
        st.success("RAG system ready!")
    else:
        vectorstore = st.session_state.vectorstore
        docs = st.session_state.docs
        embeddings = get_embedding_model()
        st.success("RAG system ready! (cached index)")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a question about the PDF...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        is_list = is_list_query(query)

        with st.spinner("Retrieving context..."):
            retrieved_docs = retrieve_documents_mmr(vectorstore, query, docs)

        with st.spinner("Compressing context..."):
            char_limit = MAX_CONTEXT_CHARS_LIST if is_list else MAX_CONTEXT_CHARS
            context = compress_context_extractive(
                query, retrieved_docs, embeddings, char_limit
            )

        with st.spinner("Generating answer..."):
            prompt = build_prompt(query, context, as_list=is_list)
            max_tokens = GROQ_MAX_TOKENS_LIST if is_list else GROQ_MAX_TOKENS
            temperature = 0.0 if is_list else 0.2
            answer = groq_complete(prompt, temperature=temperature, max_tokens=max_tokens)

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander(
                f"Sent to Groq ({len(context)} chars from "
                f"{len(retrieved_docs)} MMR chunks)"
            ):
                st.caption("Compressed context (extractive):")
                st.write(context)
                st.divider()
                st.caption("Raw retrieved chunks:")
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Chunk {i + 1}**")
                    st.write(doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": answer})
