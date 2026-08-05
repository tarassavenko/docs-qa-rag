from rag import answer_question, build_index
from config import CHUNK_SIZE, OVERLAP

with open("data/space.txt", encoding="utf-8") as f:
    text = f.read()

chunks, embedded_chunks = build_index(text, CHUNK_SIZE, OVERLAP)
result = answer_question(
    "Who was the first human went to space and when", embedded_chunks, chunks
)
print(result)
