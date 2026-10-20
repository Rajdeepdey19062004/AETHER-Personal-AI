# AETHER Personal AI

A local chat assistant built with Streamlit and Ollama. It uses the
`qwen2.5:3b` model through Ollama's local HTTP API. It can also index PDF, TXT,
Markdown, and DOCX files and find passages with semantic search.

## Project structure

```text
AETHER-Personal-AI/
├── app.py                 # Streamlit chat application
├── aether_core.py         # Text extraction and chunking
├── auth_store.py          # Local account storage
├── chat_store.py          # Chat history storage
├── document_store.py      # Local chunk and embedding index
├── embedding_engine.py    # Sentence embedding generation
├── index_notes.py         # Build the index from local documents
├── rag_engine.py          # Retrieve passages and query local Ollama
├── semantic_search.py     # Search indexed passages
├── requirements.txt       # Python dependencies
├── tests/                  # Automated tests
└── data/                   # Local-only files; excluded from Git
```

## Privacy

This repository contains application source code, not application data.
Uploaded documents, account databases, chat history, logs, and generated
document indexes are local runtime files and are excluded from version control.
Do not add real user documents, database files, credentials, or other personal
data to a commit.

## Setup (Windows PowerShell)

From this directory, create and activate the virtual environment, then install
the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Accounts and storage

Anyone with access to the app can create an account with a username and
password. Credentials and chat history are stored in `data/aether.sqlite3`.
Passwords are salted scrypt hashes, never plaintext. Each account's documents,
embeddings, and generated notes are kept in a separate folder under `data/users/`.

For remote access, deploy AETHER behind HTTPS. The local Ollama service must also
be reachable securely by the app host; do not expose Ollama's unauthenticated
port directly to the public internet.

Install and start [Ollama](https://ollama.com/), then make sure the configured
model is available:

```powershell
ollama pull qwen2.5:3b
```

Run the app from the project directory:

```powershell
python -m streamlit run app.py
```

By default, the app opens at `http://localhost:8501` and connects to Ollama at
`http://localhost:11434`. Override these with `AETHER_OLLAMA_URL` and
`AETHER_MODEL` if needed.

In the app, upload PDF, TXT, MD, or DOCX files in the sidebar and select
**Index Documents**. Then ask a question in the chat; AETHER will show the
retrieved source passages with its answer. Indexing is incremental: new files
are added, identical content is skipped using SHA-256, and changed content
under the same filename replaces its previous chunks. The sidebar lists
indexed files and requires confirmation before deleting a file and its index.
The index persists across application restarts. You can also chat with Ollama without uploading any files;
in that mode, AETHER answers general questions and says when it cannot access a
personal document.

If Sentence Transformers cannot load, indexing uses a built-in local token and
character matching fallback, so uploads and search still work. That fallback
is less semantic than the transformer model; install the requirements for
stronger meaning-based retrieval.

## Index and search notes

Place PDF, TXT, MD, or DOCX files in `data/documents/`, then run:

```powershell
python index_notes.py
python semantic_search.py
```

The index is stored in `data/index/chunks.json`, `data/index/embeddings.npy`,
and `data/index/documents.json`. Indexing preserves existing documents and
processes source files incrementally. Similarity scores describe vector
similarity; they are not probabilities that a passage is a correct answer.
