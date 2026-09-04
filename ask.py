from pathlib import Path
from rag import answer_question, build_index, reset_collection
from config import CHUNK_SIZE, OVERLAP

QUESTION = (
    "How much water is needed to produce one cup of coffee, "
    "and how much energy does global photosynthesis capture on average?"
)

collection = reset_collection()
for path in sorted(Path("data").glob("*.txt")):
    text = path.read_text(encoding="utf-8")
    collection = build_index(text, path.name, CHUNK_SIZE, OVERLAP)

result = answer_question(QUESTION, collection)

print(result["answer"], "\n")
for n, source in enumerate(result["sources"], start=1):
    print(f"[{n}] {source['source']}  score={source['score']:.3f}  id={source['id']}")
