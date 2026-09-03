import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from pathlib import Path

CHUNK_SIZE = 150
OVERLAP = 50
EMBEDDED_MODEL = "text-embedding-3-small"
LLM = "gpt-4.1-mini"
CHROMA_PATH = os.getenv("CHROMA_PATH", str(Path(__file__).parent / "chroma_db"))
COLLECTION_NAME = "documents"


load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
if __name__ == "__main__":
    if api_key and api_key.startswith("sk-proj-"):
        print("Api Key was found")
    else:
        print("Error,Api key was not found!")
