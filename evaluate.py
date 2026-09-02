import json
from pathlib import Path
from rag import retrieve, build_index
from config import CHUNK_SIZE, OVERLAP


def load_eval_set(path="eval_set.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_corpus_index(directory="data"):
    chunks = []
    embedded_chunks = []
    for path in sorted(Path(directory).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        new_chunks, new_embeddings = build_index(text, CHUNK_SIZE, OVERLAP)
        chunks.extend(new_chunks)
        embedded_chunks.extend(new_embeddings)
    return chunks, embedded_chunks


def contains_all(text, terms):
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def contains_any(text, terms):
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def check_anchors(cases, chunks):
    missing = [
        (case["id"], term)
        for case in cases
        for term in case["must_contain"]
        if not any(term.lower() in chunk.lower() for chunk in chunks)
    ]
    for case_id, term in missing:
        print(f"  WARNING [{case_id}] anchor missing from every chunk: {term!r}")
    return missing


def evaluate_retrieval(cases, chunks, embedded_chunks, k=3):
    scoreable = [case for case in cases if case["must_contain"]]
    hits = 0
    reciprocal_ranks = []

    for case in scoreable:
        retrieved = retrieve(case["question"], embedded_chunks, chunks, k=k)

        combined = "\n".join(result["text"] for result in retrieved)
        hit = contains_all(combined, case["must_contain"])

        rank = next(
            (
                position
                for position, result in enumerate(retrieved, start=1)
                if contains_any(result["text"], case["must_contain"])
            ),
            None,
        )

        if hit:
            hits += 1
            reciprocal_ranks.append(1 / rank if rank else 0.0)
        else:
            reciprocal_ranks.append(0.0)
            missing = [
                term
                for term in case["must_contain"]
                if term.lower() not in combined.lower()
            ]
            print(f"  MISS [{case['id']}] {case['question']}")
            print(f"        missing: {missing}")

    n = len(scoreable)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
    }


if __name__ == "__main__":
    cases = load_eval_set()
    chunks, embedded_chunks = build_corpus_index()

    print(
        f"indexed {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={OVERLAP})\n"
    )
    check_anchors(cases, chunks)

    scores = evaluate_retrieval(cases, chunks, embedded_chunks, k=3)

    print(f"\ncases:     {scores['n']}")
    print(f"hits:      {scores['hits']}")
    print(f"hit_rate:  {scores['hit_rate']:.1%}")
    print(f"mrr:       {scores['mrr']:.3f}")
