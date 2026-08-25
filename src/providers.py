import os

from dotenv import load_dotenv
from langchain_postgres import PGVector

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()

def _require(var):
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"{var} não está definida no .env")
    return value

def get_embeddings():
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        _require("OPENAI_API_KEY")
        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )

    if PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _require("GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        )

    raise RuntimeError(f"LLM_PROVIDER inválido: '{PROVIDER}'. Use 'openai' ou 'gemini'.")

def get_llm():
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        _require("OPENAI_API_KEY")
        return ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"))

    if PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        _require("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
            temperature=0,
        )

    raise RuntimeError(f"LLM_PROVIDER inválido: '{PROVIDER}'. Use 'openai' ou 'gemini'.")

def get_vector_store(pre_delete_collection=False):
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME", "documentos"),
        connection=_require("DATABASE_URL"),
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )