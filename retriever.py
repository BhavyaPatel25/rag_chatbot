import os
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DOC_PATH = os.path.join("data", "Bhavya Patel Resume.docx")
VECTOR_DIR = "vectorstore"

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

# IMPORTANT: Do NOT call persist() anywhere
if not os.path.exists(VECTOR_DIR):
    loader = Docx2txtLoader(DOC_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )
else:
    vectordb = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings
    )

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

def retrieve_context(question: str) -> str:
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)
