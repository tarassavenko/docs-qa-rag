import json
from pathlib import Path
from rag import retrieve, build_index, reset_collection, retrieve_hybrid
from config import CHUNK_SIZE, OVERLAP, EMBEDDED_MODEL, TOP_K, RETRIEVAL_MODE
from datetime import datetime, timezone
from keyword_search import build_keyword_index
import argparse


def load_eval_set(path="eval_set.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_corpus_index(directory="data", chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    collection = reset_collection()
    for path in sorted(Path(directory).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        collection = build_index(text, path.name, chunk_size, overlap)

    return collection


def contains_all(text, terms):
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def contains_any(text, terms):
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def check_anchors(cases, collection):
    documents = collection.get()["documents"]
    missing = [
        (case["id"], term)
        for case in cases
        for term in case["must_contain"]
        if not any(term.lower() in chunk.lower() for chunk in documents)
    ]
    for case_id, term in missing:
        print(f"  WARNING [{case_id}] anchor missing from every chunk: {term!r}")
    return missing


def evaluate_retrieval(
    cases, collection, k=TOP_K, keyword_index=None, mode=RETRIEVAL_MODE
):
    scoreable = [case for case in cases if case["must_contain"]]
    hits = 0
    reciprocal_ranks = []
    failed_ids = []

    for case in scoreable:
        if mode == "vector":
            retrieved = retrieve(case["question"], collection, k=k)
        elif mode == "hybrid":
            retrieved = retrieve_hybrid(
                case["question"], collection, keyword_index, k=k
            )
        else:
            raise ValueError(f"unknown retrieval mode: {mode!r}")

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
            failed_ids.append(case["id"])
            print(f"  MISS [{case['id']}] {case['question']}")
            print(f"        missing: {missing}")

    n = len(scoreable)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "failed_ids": failed_ids,
    }


def build_parser():

    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against the eval set."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K,
        help="number of chunks to retrieve per question (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="words per chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help="overlapping words between chunks (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["vector", "hybrid"],
        default=RETRIEVAL_MODE,
        help="choosing the retrieval mode (default: %(default)s)",
    )

    return parser


def log_result(record, path="eval_results.jsonl"):
    with open(path, encoding="utf-8", mode="a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    args = build_parser().parse_args()
    cases = load_eval_set()
    collection = build_corpus_index(chunk_size=args.chunk_size, overlap=args.overlap)
    if args.mode == "hybrid":
        keyword_index = build_keyword_index(collection)
    else:
        keyword_index = None

    print(
        f"indexed {collection.count()} chunks "
        f"(chunk_size={args.chunk_size}, overlap={args.overlap}, k={args.k},mode={args.mode})\n"
    )
    missing_anchors = check_anchors(cases, collection)
    scores = evaluate_retrieval(
        cases, collection, k=args.k, keyword_index=keyword_index, mode=args.mode
    )
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "k": args.k,
        "mode": args.mode,
        "embedding_model": EMBEDDED_MODEL,
        "chunk_count": collection.count(),
        **scores,
        "missing_anchors": missing_anchors,
    }
    log_result(record)

    print(f"\ncases:     {scores['n']}")
    print(f"hits:      {scores['hits']}")
    print(f"hit_rate:  {scores['hit_rate']:.1%}")
    print(f"mrr:       {scores['mrr']:.3f}")
