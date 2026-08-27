# docs-qa-rag

A question-answering backend over your own documents, built with a hand-rolled
Retrieval-Augmented Generation (RAG) pipeline.

Ask a question about an ingested document, and the system retrieves the most
relevant passages by semantic similarity and asks an LLM to answer using only
those passages.

## How it works

1. **Chunk** — the source document is split into overlapping word-based windows.
2. **Embed** — each chunk is turned into a vector via OpenAI's embeddings API.
3. **Retrieve** — the question is embedded and scored against every chunk with
   cosine similarity; the top-k chunks are returned.
4. **Generate** — the retrieved chunks are assembled into a prompt and sent to a
   chat model, which is instructed to answer only from the provided context.

## Status

Work in progress. The pipeline is served as a FastAPI application: documents are
ingested over HTTP, chunked, embedded and held in memory, and questions are
answered by a chat model constrained to the retrieved context. A real vector
store, per-document metadata, and an evaluation harness are planned.

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
- **In-memory index.** Chunks and their embeddings are held in memory and
  rebuilt on each run. A persistent vector store replaces this later.
- **The index is built at startup, not at import.** A FastAPI lifespan handler
  seeds the index and stores it on `app.state`, so importing the module — for a
  test, or to inspect the routes — does not trigger a full embedding run, and
  there is a matching place to put teardown when a vector store needs closing.
- **Ingestion appends rather than replaces.** `POST /ingest` extends the index
  with a new document's chunks. This is the behaviour multi-document retrieval
  needs, but see the open question below: chunks do not yet carry any record of
  which document they came from.
- **Index building is separate from answering.** `build_index` is called once by
  the caller and its result is passed into `answer_question`, so embedding the
  document does not happen per question. This is the shape an API server needs:
  build at startup, query per request.
- **Sources are labelled and cited.** Retrieved chunks are wrapped in numbered
  `<source>` tags and the model is asked to cite the ids it used, which makes it
  possible to tell a retrieval failure (wrong chunk fetched) from a generation
  failure (right chunk, wrong answer).

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

Generation runs at `temperature=0`, since the task is extraction from supplied
text rather than open-ended writing.

## Open questions

Recorded as they come up, and intended to be settled with measurements against
an evaluation set rather than by intuition.

- **Chunk size and overlap are untuned.** The current values are a placeholder.
  Chunks that are too small fragment a single fact across several of them;
  chunks that are too large dilute the embedding across unrelated topics.
- **`k` is untuned.** It trades off against chunk size — larger chunks mean
  fewer are needed to cover an answer.
- **Retrieval quality is unmeasured.** Vector search alone may not be enough;
  hybrid keyword search and a re-ranking step are worth testing once there is
  an eval set to compare against.
- **Citation accuracy is unverified.** The model cites source ids, but whether
  those ids are the chunks it actually relied on has not been checked.
- **The test document is small.** At roughly 1,000 words it produces only a
  handful of chunks, so top-k retrieval selects a large fraction of the whole
  document. Retrieval experiments will not produce meaningful numbers until the
  corpus is large enough for retrieval to be genuinely selective.
- **Chunks carry no source identity.** Ingestion appends to two parallel lists,
  so once several documents are indexed there is no way to say which document a
  retrieved chunk came from, to cite a document by name, or to remove a single
  document without restarting the process. Attaching metadata at insert time is
  what a vector store's record model provides.
- **The index does not survive a restart.** Everything is rebuilt from the seed
  document on startup, so any document added through `/ingest` is lost.

## Layout

| File | Purpose |
| --- | --- |
| `chunking.py` | Splits text into overlapping word-based chunks |
| `embeddings.py` | Wraps the OpenAI embeddings API (single and batch) |
| `similarity.py` | Cosine similarity between two vectors |
| `config.py` | API client, model names, and chunking constants |
| `rag.py` | Index building, top-k retrieval, prompt assembly, generation |
| `main.py` | FastAPI application — the service itself |
| `ask.py` | Script entry point for exercising the pipeline without HTTP |

`rag.py` imports no web framework, and `main.py` contains no retrieval logic.
That separation is why the same pipeline can be driven by a server, a script, or
an evaluation harness without duplicating anything that matters.

## Setup

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in a `.env` file in the project root (see `.env.example`),
then start the server:

```bash
fastapi dev main.py
```

To run a single question through the pipeline without starting a server:

```bash
python ask.py
```
