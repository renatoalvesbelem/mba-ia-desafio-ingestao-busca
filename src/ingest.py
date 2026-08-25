import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import get_vector_store

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")

def resolve_pdf():
    pdf = Path(PDF_PATH)
    if not pdf.is_absolute():
        pdf = BASE_DIR / pdf
    if not pdf.exists():
        raise SystemExit(f"PDF não encontrado em {pdf}")
    return pdf

def ingest_pdf():
    pdf = resolve_pdf()

    pages = PyPDFLoader(str(pdf)).load()
    print(f"{len(pages)} páginas lidas de {pdf.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)
    print(f"{len(chunks)} chunks gerados")

    store = get_vector_store(pre_delete_collection=True)
    store.add_documents(chunks)

    print("Ingestão concluída.")

if __name__ == "__main__":
    ingest_pdf()