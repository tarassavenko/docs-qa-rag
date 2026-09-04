# docs-qa-rag

A question-answering backend over your own documents, built with a hand-rolled
Retrieval-Augmented Generation (RAG) pipeline.

Ask a question about an ingested document, and the system retrieves the most
relevant passages by semantic similarity and asks an LLM to answer using only
those passages.

## How it works

1. **Chunk** — the source document is split into overlapping word-based windows.
2. **Embed** — each chunk is turned into a vector via OpenAI's embeddings API.
3. **Store** — chunks, vectors and their source metadata are written to a Chroma
   collection configured for cosine distance.
4. **Retrieve** — the question is embedded and the collection returns the top-k
   nearest chunks, each with its similarity score and originating document.
5. **Generate** — the retrieved chunks are assembled into a prompt and sent to a
   chat model, which is instructed to answer only from the provided context and
   to cite the sources it used.

## Status

Work in progress. The pipeline is served as a FastAPI application backed by a
persistent Chroma collection: documents are ingested over HTTP, chunked,
embedded and stored with per-document metadata, and questions are answered by a
chat model constrained to the retrieved context and asked to cite its sources.
Retrieval quality is measured against a hand-written evaluation set. Retrieval
tuning, richer answer-quality metrics and deployment are planned.

## API

Run the server with `fastapi dev main.py`, then open `/docs` for an interactive
UI generated from the request and response models.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service name and a pointer to the docs |
| `GET` | `/health` | Liveness check — is the process running |
| `GET` | `/status` | Readiness check — is anything indexed, and how many chunks |
| `POST` | `/ask` | Answer a question from the indexed documents, with sources |
| `POST` | `/ingest` | Add a document's text to the index |

`/health` and `/status` are deliberately separate: the first answers "is the
process alive", the second "can it actually serve a request". A server with no
document indexed is healthy but not ready.

### Error handling

| Status | Meaning |
| --- | --- |
| `422` | Request failed validation — empty or oversized question or document |
| `502` | The upstream embedding or chat request failed |
| `503` | Nothing has been indexed yet, so `/ask` has nothing to answer from |

Input constraints live on the Pydantic models rather than in handler code, so
FastAPI rejects bad input before it reaches the pipeline and documents the
limits in `/docs`. Failures of the upstream model are logged with their full
traceback server-side, while the client receives a generic message — the
operator needs the detail, the caller does not.

## Design decisions

- **No RAG framework.** LangChain, LlamaIndex and similar libraries are
  deliberately avoided. Chunking, similarity search, ranking and prompt
  assembly are implemented directly so that the behaviour of each stage — and
  its failure modes — is fully understood rather than abstracted away.
  A LangChain implementation of the same `answer_question` interface is planned
  as a second module, so that both can be run against the same evaluation set
  and compared directly on retrieval and answer quality.
- **Plain text only.** Ingestion is scoped to `.txt` and `.md`. PDF extraction
  is a parsing problem rather than a retrieval problem and adds little to the
  goals of this project, so it is intentionally out of scope.
- **Chroma for storage, but embeddings stay hand-rolled.** Chroma computes
  vectors itself if asked, using a local sentence-transformers model. Passing
  our own OpenAI vectors instead keeps the embedding model an explicit choice,
  keeps batching and retries in our code, and — during the migration — meant
  storage was the only variable that changed, so the evaluation numbers stayed
  comparable. Chroma stores and searches; it does not chunk or embed.
- **Cosine distance, converted to similarity at the boundary.** The collection
  is created with `hnsw:space: cosine`, since the default is L2 squared. Chroma
  returns distances where lower is better, so `retrieve` returns `1 - distance`
  and callers keep the higher-is-better score they always had. Getting this
  backwards retrieves the *least* relevant chunks and still produces fluent
  answers, which is precisely the failure the evaluation set exists to catch.
- **Deterministic chunk ids.** Ids are `"{document}-{n}"` rather than UUIDs, so
  re-indexing a document overwrites its chunks in place instead of duplicating
  them, and a source id in an API response says which document it came from
  without a lookup.
- **The index is built at startup, not at import.** A FastAPI lifespan handler
  resets the collection and indexes `data/`, storing the collection on
  `app.state`. Importing the module — for a test, or to inspect the routes —
  does not trigger a full embedding run.
- **Ingestion appends rather than replaces.** `POST /ingest` adds a new
  document's chunks to the collection alongside the existing ones, tagged with
  the caller-supplied `source`. Per-document deletion is not implemented yet,
  but the metadata needed for it is now stored.
- **Index building is separate from answering.** `build_index` is called once by
  the caller and its result is passed into `answer_question`, so embedding the
  document does not happen per question. This is the shape an API server needs:
  build at startup, query per request.
- **Sources are labelled and cited.** Retrieved chunks are wrapped in
  `<source id="1" document="coffee.txt">` tags, and the model is asked to cite
  the ids it used and name the document. This makes it possible to tell a
  retrieval failure (wrong chunk fetched) from a generation failure (right
  chunk, wrong answer), and to attribute an answer that draws on several
  documents.
- **`similarity.py` is retained but unused.** Cosine similarity is now computed
  inside Chroma. The hand-written implementation is kept because it documents
  what the vector store was brought in to replace.

## Grounding behaviour

The system prompt is what keeps answers tied to the retrieved context. Three
cases were tested against a document about outer space:

| Case | Question | Result |
| --- | --- | --- |
| Answerable | Covered directly by the document | Answered with citations |
| Unanswerable | "How long does astronaut training take?" | Exact refusal string |
| Partial | Treaty contents (present) + signatories (absent) | Answered the supported half, named the gap |

Two findings worth recording:

- A naive prompt (*"answer based on the context"*) did **not** hallucinate — it
  refused. But it refused in free-form prose and then offered to answer from
  outside knowledge, which is an escape hatch in a multi-turn setting.
- Specifying an exact refusal sentence and an explicit prohibition on prior
  knowledge produced a deterministic, string-matchable refusal. That matters
  less for answer quality than for evaluation: "did it refuse?" becomes an exact
  comparison rather than a judgement call.
- **Conditional instructions are effectively optional.** A rule reading *"name
  the document when it matters"* was ignored even on an answer drawn from two
  documents — the model judged that it did not matter. Rewriting it as an
  unconditional instruction produced the behaviour immediately. The same pattern
  as the refusal string: if it must happen every time, the prompt has to say so
  without a hedge.

Generation runs at `temperature=0`, since the task is extraction from supplied
text rather than open-ended writing.

## Evaluation

`eval_set.json` holds 24 hand-written questions over the corpus — 17 answerable
(two of which require chunks from two different documents), 4 unanswerable, and
3 partially answerable. Each scoreable question is anchored to distinctive
phrases the correct chunk must contain, rather than to a chunk id, so the set
survives re-chunking. `evaluate.py` reports the fraction of questions whose
top-k chunks collectively contain every anchor, plus mean reciprocal rank.

It also warns when an anchor no longer appears in any chunk — which is what a
change in chunk size looks like from the outside, and would otherwise be
misread as a retrieval regression.

Baseline at `chunk_size=150`, `overlap=50`, `k=3`, over 416 chunks:

| Index | Hit rate | MRR |
| --- | --- | --- |
| In-memory lists + hand-written cosine similarity | 85.0% | 0.817 |
| Chroma collection (cosine) | 85.0% | 0.817 |

Identical numbers across the storage migration is the point of having built the
evaluation set first: it demonstrates the swap preserved retrieval quality
rather than asserting it.

The three failures are more useful than the score, because each points at a
different fix:

| Question | Failure | What it suggests |
| --- | --- | --- |
| Who discovered photosynthesis? | The chunk containing *Jan Ingenhousz* is dominated by unrelated material, diluting its embedding | Smaller chunks |
| When was Roman citizenship granted to all freeborn inhabitants? | The corpus states this almost verbatim, but it loses to other chunks in a 15,000-word document of similar-sounding imperial history | Hybrid keyword search — *Caracalla* is a rare token that exact matching would find instantly |
| Term coined + Constantinople fell | Cross-document; the machine-learning half took all three slots | Larger `k` |

Citation accuracy was spot-checked on the cross-document questions: for the
coffee/photosynthesis question the model cited `[1][3]`, which were exactly the
two chunks carrying the two facts, and ignored the third retrieved chunk.

## Open questions

Recorded as they come up, and intended to be settled with measurements against
an evaluation set rather than by intuition.

- **Chunk size and overlap are untuned.** The current values are a placeholder,
  and the evaluation set now makes a sweep measurable rather than a guess.
  Chunks that are too small fragment a single fact across several of them;
  chunks that are too large dilute the embedding across unrelated topics — the
  photosynthesis failure above is the second case.
- **`k` is untuned.** It trades off against chunk size, and against how many
  documents an answer needs to span.
- **Vector search alone may not be enough.** The Caracalla failure is a near
  verbatim match that pure semantic similarity still ranks below three other
  chunks. Hybrid keyword search (BM25 alongside vectors) and a re-ranking step
  are the obvious next experiments.
- **Chunking ignores document structure.** `chunk_text` splits on word count
  regardless of paragraph or sentence boundaries, and normalises whitespace
  when rejoining. Recursive splitting on paragraph breaks would produce more
  coherent chunks and fewer anchors split across a boundary; it is untested.
- **Ingested documents are lost on restart.** The Chroma collection persists to
  disk, but the lifespan handler resets it and rebuilds from `data/` on every
  start, so anything added through `/ingest` disappears. Reusing an existing
  collection would fix this but raises a staleness question that has not been
  worked out.
- **Per-document deletion is not implemented.** The metadata to support it is
  stored, but there is no endpoint to remove or replace a single document.
- **The evaluation set is hand-written and small.** Twenty scoreable questions
  cannot cover a corpus of 416 chunks. Synthetic generation would give volume,
  but questions generated *from* a chunk are answerable by that chunk by
  construction, so they systematically over-report retrieval quality. Running
  both and comparing the two numbers is the interesting experiment.

## Layout

| File | Purpose |
| --- | --- |
| `chunking.py` | Splits text into overlapping word-based chunks |
| `embeddings.py` | Wraps the OpenAI embeddings API (single and batch) |
| `similarity.py` | Hand-written cosine similarity — superseded by Chroma, kept for reference |
| `config.py` | API clients, model names, chunking constants, Chroma settings |
| `rag.py` | Collection management, indexing, retrieval, prompt assembly, generation |
| `main.py` | FastAPI application — the service itself |
| `ask.py` | Script entry point for exercising the pipeline without HTTP |
| `evaluate.py` | Retrieval evaluation — hit rate and MRR against the eval set |
| `eval_set.json` | 24 hand-written questions with expected-content anchors |
| `data/` | Sample corpus (~41,000 words across five documents) |

`rag.py` imports no web framework, and `main.py` contains no retrieval logic.
That separation is why the same pipeline can be driven by a server, a script, or
an evaluation harness without duplicating anything that matters — and why
swapping the storage layer for Chroma changed two function bodies rather than
the whole codebase.

## Setup

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in a `.env` file in the project root (see `.env.example`),
then start the server:

```bash
fastapi dev main.py
```

The server indexes everything in `data/` at startup, writing the Chroma
collection to `chroma_db/` (configurable with `CHROMA_PATH`).

To run a single question through the pipeline without starting a server:

```bash
python ask.py
```

To measure retrieval quality against the evaluation set:

```bash
python evaluate.py
```
