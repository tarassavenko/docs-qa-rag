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

Work in progress. Retrieval (chunking, embedding, top-k ranking) is implemented;
prompt building and generation are next. A FastAPI layer, a real vector store,
and an evaluation harness are planned.

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

## Layout

| File | Purpose |
| --- | --- |
| `chunking.py` | Splits text into overlapping word-based chunks |
| `embeddings.py` | Wraps the OpenAI embeddings API (single and batch) |
| `similarity.py` | Cosine similarity between two vectors |
| `rag.py` | Index building and top-k retrieval |
| `main.py` | Entry point: loads a document and answers a question |

## Setup

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in a `.env` file in the project root, then:

```bash
python main.py
```
