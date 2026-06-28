import hashlib
import os
from glob import glob

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
MANIFEST_PATH = os.path.join(VECTOR_DIR, ".source_manifest")

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()


def _build_embeddings():
    """Embeddings are pluggable so chat (Ollama Cloud) and embeddings can differ.

    OpenAI text-embedding-3-large is a hosted API that works from anywhere
    (incl. Render) with no local model - good default for production.
    Ollama embeddings stay available for fully-local runs.
    """
    provider = EMBEDDINGS_PROVIDER
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        host = os.getenv("EMBED_HOST", os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        api_key = os.getenv("EMBED_API_KEY", os.getenv("OLLAMA_API_KEY", ""))
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(
            base_url=host,
            model=model,
            client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}} if api_key else {},
        ), f"ollama:{host}|{model}"

    # default: openai
    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model=model), f"openai:{model}"


embeddings, EMBED_IDENTITY = _build_embeddings()


def _load_documents():
    """Load every supported source file in data/ into one document list."""
    docs = []
    for path in sorted(glob(os.path.join(DATA_DIR, "*"))):
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".docx":
                docs.extend(Docx2txtLoader(path).load())
            elif ext == ".pdf":
                docs.extend(PyPDFLoader(path).load())
            elif ext in (".md", ".txt"):
                docs.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as exc:  # pragma: no cover
            print(f"[retriever] failed to load {path}: {exc}")
    return docs


def _source_manifest() -> str:
    """Hash of source files + sizes + mtimes AND the embedding identity.

    Embedding identity is included so the index auto-rebuilds when you switch
    embedding models (vectors from different models are not comparable).
    """
    entries = []
    for path in sorted(glob(os.path.join(DATA_DIR, "*"))):
        if os.path.isfile(path):
            st = os.stat(path)
            entries.append(f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256(("|".join(entries) + "||embed=" + EMBED_IDENTITY).encode()).hexdigest()


def _build_vectorstore() -> Chroma:
    docs = _load_documents()
    if not docs:
        raise RuntimeError(f"No documents found in {DATA_DIR}/")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=140,
        add_start_index=True,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR,
    )
    os.makedirs(VECTOR_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        fh.write(_source_manifest())
    print(f"[retriever] built vectorstore from {len(docs)} doc(s) -> {len(chunks)} chunks ({EMBED_IDENTITY})")
    return vectordb


def _load_or_rebuild() -> Chroma:
    if not os.path.exists(VECTOR_DIR) or not os.path.exists(MANIFEST_PATH):
        return _build_vectorstore()

    try:
        with open(MANIFEST_PATH, encoding="utf-8") as fh:
            saved = fh.read().strip()
    except OSError:
        return _build_vectorstore()

    if saved != _source_manifest():
        print("[retriever] sources or embedding model changed -> rebuilding vectorstore")
        return _build_vectorstore()

    return Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)


vectordb = _load_or_rebuild()

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)


def retrieve_context(question: str) -> str:
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)
