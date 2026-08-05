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

Work in progress. The pipeline runs end to end as a script: a document is
chunked, embedded, queried by cosine similarity, and answered by a chat model
constrained to the retrieved context. A FastAPI layer, a real vector store, and
an evaluation harness are planned.

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

## Layout

| File | Purpose |
| --- | --- |
| `chunking.py` | Splits text into overlapping word-based chunks |
| `embeddings.py` | Wraps the OpenAI embeddings API (single and batch) |
| `similarity.py` | Cosine similarity between two vectors |
| `config.py` | API client, model names, and chunking constants |
| `rag.py` | Index building, top-k retrieval, prompt assembly, generation |
| `main.py` | Entry point: loads a document, builds the index, asks a question |

## Setup

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in a `.env` file in the project root, then:

```bash
python main.py
```
