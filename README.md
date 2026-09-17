# docs-qa-rag

A question-answering backend over your own documents, built with a hand-rolled
Retrieval-Augmented Generation (RAG) pipeline.

Ask a question about the indexed documents, and the system retrieves the most
relevant passages — combining semantic search with keyword search — and asks an
LLM to answer using only those passages, citing the sources it used.

## How it works

1. **Chunk** — each document is split into overlapping word-based windows.
2. **Embed** — each chunk is turned into a vector via OpenAI's embeddings API.
3. **Store** — chunks, vectors and their source metadata are written to a Chroma
   collection configured for cosine distance. A BM25 keyword index is built in
   memory from the same chunks.
4. **Retrieve** — the question is run through both vector search and BM25
   keyword search, 20 candidates each. The two ranked lists are merged with
   reciprocal rank fusion and the top 5 chunks are kept.
5. **Generate** — the retrieved chunks are assembled into a prompt and sent to a
   chat model, which is instructed to answer only from the provided context and
   to cite the sources it used.

## Status

Work in progress. The pipeline is served as a FastAPI application backed by a
persistent Chroma collection and an in-memory keyword index. Documents are
ingested over HTTP and questions are answered with hybrid retrieval and cited
generation. Retrieval has been tuned and measured against a hand-written
evaluation set: hit rate went from 85% to 100% through a chunking and `k` sweep
followed by hybrid search. Deployment is next.

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

In hybrid mode, the `score` on each source returned by `/ask` is the reciprocal
rank fusion score (roughly 0.016–0.033), not a cosine similarity. A chunk found
near the top of both searches scores around 0.03; a chunk found by only one
scores around 0.016.

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
  deliberately avoided. Chunking, similarity search, ranking, fusion and prompt
  assembly are implemented directly so that the behaviour of each stage — and
  its failure modes — is fully understood rather than abstracted away.
  A LangChain implementation of the same `answer_question` interface is planned
  as a second module, so that both can be run against the same evaluation set
  and compared directly on retrieval and answer quality.
- **Plain text only.** Ingestion is scoped to `.txt` and `.md`. PDF extraction
  is a parsing problem rather than a retrieval problem and adds little to the
  goals of this project, so it is intentionally out of scope.
- **Hybrid retrieval, fused by rank rather than by score.** Embeddings are good
  at meaning and weak at rare exact terms; BM25 is the opposite. Their scores
  cannot simply be added — cosine similarity is bounded, a BM25 score is not —
  so the two lists are merged with reciprocal rank fusion (Cormack, Clarke &
  Büttcher, 2009): each chunk scores `1 / (60 + rank)` for every list it
  appears in, and the totals are sorted. Each method contributes 20 candidates
  before fusion so that a chunk ranked modestly by one method can still be
  rescued by the other. Pure vector retrieval remains available through
  `RETRIEVAL_MODE` and `evaluate.py --mode vector`, so the two can always be
  compared.
- **The keyword index is derived from Chroma, not from the files.** Building
  BM25 from `collection.get()` guarantees it sees exactly the same chunks, with
  the same ids, as vector search — which is what lets fusion join the two lists
  on chunk id. It lives in memory, costs no API calls to build, and is rebuilt
  after every `POST /ingest`; without the rebuild, newly ingested documents
  would be reachable only through vector search.
- **Tokenisation stems and drops stopwords.** BM25 matches words, so query and
  documents both go through the same NLTK pipeline: case folding, Porter
  stemming and English stopword removal. *Granted* and *granting* both reduce
  to *grant*, and words like *when* and *the* contribute nothing.
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
  them, a source id in an API response says which document it came from
  without a lookup, and the two retrieval methods can be joined on id.
- **The index is built at startup, not at import.** A FastAPI lifespan handler
  resets the collection, indexes `data/` and builds the keyword index, storing
  both on `app.state`. Importing the module — for a test, or to inspect the
  routes — does not trigger a full embedding run.
- **Ingestion appends rather than replaces.** `POST /ingest` adds a new
  document's chunks to the collection alongside the existing ones, tagged with
  the caller-supplied `source`. Per-document deletion is not implemented yet,
  but the metadata needed for it is stored.
- **Index building is separate from answering.** Indexes are built once by the
  caller and passed into `answer_question`, so embedding the document does not
  happen per question. This is the shape an API server needs: build at startup,
  query per request.
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
cases were first tested against a single document about outer space:

| Case | Question | Result |
| --- | --- | --- |
| Answerable | Covered directly by the document | Answered with citations |
| Unanswerable | "How long does astronaut training take?" | Exact refusal string |
| Partial | Treaty contents (present) + signatories (absent) | Answered the supported half, named the gap |

Findings:

- A naive prompt (*"answer based on the context"*) did **not** hallucinate — it
  refused. But it refused in free-form prose and then offered to answer from
  outside knowledge, which is an escape hatch in a multi-turn setting.
- Specifying an exact refusal sentence and an explicit prohibition on prior
  knowledge produced a string-matchable refusal. That matters less for answer
  quality than for evaluation: "did it refuse?" becomes an exact comparison
  rather than a judgement call.
- **Conditional instructions are effectively optional.** A rule reading *"name
  the document when it matters"* was ignored even on an answer drawn from two
  documents — the model judged that it did not matter. Rewriting it as an
  unconditional instruction produced the behaviour immediately. The same pattern
  as the refusal string: if it must happen every time, the prompt has to say so
  without a hedge.
- **More context loosens refusals.** Raising `k` from 3 to 5 made refusals on
  near-miss questions wordier: the model gives the refusal sentence and then
  describes related material it did find. For *"How does drinking coffee affect
  blood pressure?"* the answer was the exact refusal in 6 of 6 runs at `k=3`
  but only 1 of 5 at `k=5`, because the extra chunks cover coffee and
  cardiovascular health. No run invented an answer. A stricter wording of the
  refusal rule fixed one such question but not the other, and made partial
  answers wordier, so it was not adopted.
- **`temperature=0` is not deterministic.** The same question at the same
  settings occasionally switched between an exact and a verbose refusal, so
  exact-match refusal detection was never fully reliable.
- **Hybrid retrieval turned one "unanswerable" question into a partial one.**
  For *"What kind of espresso machine produces the best crema?"*, keyword search
  surfaced a chunk stating that robusta beans give a better crema. The model
  then answered — in 3 of 3 runs — with what the sources do support and stated
  that no machine type is identified. That is correct grounded behaviour, but it
  contains no refusal sentence, so the question's label is borderline.

Generation runs at `temperature=0`, since the task is extraction from supplied
text rather than open-ended writing.

## Evaluation

`eval_set.json` holds 24 hand-written questions over the corpus — 17 answerable
(two of which require chunks from two different documents), 4 unanswerable, and
3 partially answerable. Each scoreable question is anchored to distinctive
phrases the correct chunk must contain, rather than to a chunk id, so the set
survives re-chunking. `evaluate.py` reports the fraction of the 20 scoreable
questions whose top-k chunks collectively contain every anchor (hit rate), plus
mean reciprocal rank.

It also warns when an anchor no longer appears in any chunk — which is what a
change in chunk size looks like from the outside, and would otherwise be
misread as a retrieval regression. No run in the experiments below had a
missing anchor, so every number is comparable.

Configuration comes from command-line flags, and every run appends one JSON
record — config, metrics, failed question ids, missing anchors — to
`eval_results.jsonl`, so the history of experiments is kept alongside the code.

```bash
python evaluate.py --chunk-size 300 --overlap 60 --k 3 --mode vector
```

### Storage migration

Moving from in-memory lists to Chroma, at `chunk_size=150`, `overlap=50`,
`k=3`, over 416 chunks:

| Index | Hit rate | MRR |
| --- | --- | --- |
| In-memory lists + hand-written cosine similarity | 85.0% | 0.817 |
| Chroma collection (cosine) | 85.0% | 0.817 |

Identical numbers across the migration is the point of having built the
evaluation set first: it demonstrates the swap preserved retrieval quality
rather than asserting it.

### Chunk size and overlap

Vector retrieval, `k=3`:

| Chunk size | Overlap | Chunks | Hit rate | MRR |
| --- | --- | --- | --- | --- |
| 50 | 10 | 1038 | 85% | 0.667 |
| 100 | 20 | 520 | 75% | 0.617 |
| 150 | 0 | 279 | 75% | 0.642 |
| 150 | 30 | 348 | 75% | 0.700 |
| **150** | **50** | **416** | **85%** | **0.817** |
| 150 | 75 | 555 | 80% | 0.625 |
| 300 | 60 | 175 | 85% | 0.717 |
| 300 | 100 | 209 | 85% | 0.758 |
| 300 | 150 | 279 | 80% | 0.692 |
| 600 | 120 | 89 | 75% | 0.667 |

Hit rate only moved between 75% and 85%. With 20 scoreable questions, one
question is 5 points, so most of this spread is noise, and 150/50 looks more
like an isolated peak than a trend — both of its overlap neighbours scored
lower. Two questions failed in every configuration, which ruled out chunking as
their cause.

### Number of retrieved chunks

| Configuration | k=1 | k=3 | k=5 | k=10 |
| --- | --- | --- | --- | --- |
| 150 / 50 | 75% · 0.750 | 85% · 0.817 | **90% · 0.827** | 90% · 0.827 |
| 300 / 100 | 70% · 0.700 | 85% · 0.758 | 90% · 0.808 | 90% · 0.808 |

`k` mattered more than chunking: going from 3 to 5 fixed two questions in both
configurations, and `k=10` produced identical results for twice the context.
150/50 at `k=5` was kept over 300/100 because it has the higher MRR and sends
roughly half as much text to the model on every request; the extra chunks it
produces are an indexing cost paid once.

This metric measures recall only. A larger `k` can only raise the hit rate; it
cannot show whether the additional chunks are noise — which is exactly what
showed up in the refusal behaviour above.

### Hybrid retrieval

`chunk_size=150`, `overlap=50`, `k=5`:

| Mode | Hit rate | MRR | Failed questions |
| --- | --- | --- | --- |
| Vector | 90% | 0.827 | 14, 17 |
| Hybrid (BM25 + RRF) | **100%** | **0.900** | none |

Both remaining failures were cases where the right chunk existed but sat just
outside vector search's reach:

- **Roman citizenship.** The chunk stating that citizenship was granted *"during
  the reign of Caracalla"* ranked 18th by vector similarity, among many
  similar-sounding passages of imperial history. BM25 ranked it 2nd on the rare
  words *citizenship*, *freeborn* and *inhabitants*, and fusion placed it 4th.
- **Term coined + Constantinople fell.** The chunk containing *1453* ranked 14th
  by vector similarity — the question's embedding is dominated by its
  machine-learning half. BM25 ranked it 4th on the rare token *Constantinople*,
  and fusion placed it 3rd.

### Diagnoses versus outcomes

| Question | Original diagnosis | What actually happened |
| --- | --- | --- |
| Who discovered photosynthesis? | Diluted chunk — needs smaller chunks | Smaller chunks helped at some sizes but not consistently; `k=5` fixed it reliably |
| Roman citizenship granted to all freeborn inhabitants | Rare tokens — needs keyword search | Fixed by hybrid retrieval, as predicted |
| Water per cup of coffee + energy from photosynthesis | — (passed at the 150/50 baseline) | Failed in 9 of 10 chunking configurations at `k=3`; stable only from `k=5` |
| Term coined + Constantinople fell | Cross-document — needs larger `k`, then query decomposition | Both wrong: it still failed at `k=10`. The chunk was vector rank 14 and keyword search was enough to surface it |

Recording the diagnoses that turned out wrong is deliberate. The two questions
that "needed" a particular fix were each resolved by something else, which is
the argument for measuring changes rather than reasoning about them.

### Limitations of these numbers

- **The evaluation set is at its ceiling.** At 100%, it can no longer show an
  improvement, and a small regression only appears if it happens to hit one of
  20 questions.
- **Twenty questions give coarse measurements.** Each question is worth 5
  points, so differences of one or two questions between configurations are
  within noise.
- **It measures retrieval only.** Whether an answer is correct and grounded is
  checked by hand on selected questions, not measured.

Citation accuracy was spot-checked on the cross-document questions: for the
coffee/photosynthesis question the model cited `[1][3]`, which were exactly the
two chunks carrying the two facts, and ignored the other retrieved chunks.

## Open questions

Recorded as they come up, and intended to be settled with measurements rather
than by intuition.

- **The evaluation set needs to be harder and larger.** It is now saturated.
  Synthetic generation would give volume, but questions generated *from* a
  chunk are answerable by that chunk by construction, so they systematically
  over-report retrieval quality. Running both and comparing the two numbers is
  the interesting experiment.
- **Refusal rate is not measured.** Exact or prefix matching on the refusal
  sentence is brittle given the nondeterminism and the borderline crema
  question above; judging refusals with a model is likely necessary.
- **Fusion margins are thin.** The Roman citizenship chunk reaches the top 5 by
  0.0001, and only because its vector rank (18) falls inside the 20-candidate
  pool. Weighted fusion and a larger candidate pool are untested.
- **Near-miss chunks weaken refusals.** A minimum retrieval score before
  generation might filter chunks that are related but not relevant; untested.
- **Re-ranking is untested.** Retrieving a larger pool and re-ranking it with a
  cross-encoder or a model call is the usual next step after hybrid search.
- **Chunking ignores document structure.** `chunk_text` splits on word count
  regardless of paragraph or sentence boundaries. The sweep suggests this has
  little effect on hit rate for this corpus, and the failures it left were
  ranking problems rather than boundary problems, but recursive splitting on
  paragraph breaks is untested.
- **Ingested documents are lost on restart.** The Chroma collection persists to
  disk, but the lifespan handler resets it and rebuilds from `data/` on every
  start, so anything added through `/ingest` disappears. Reusing an existing
  collection would fix this but raises a staleness question that has not been
  worked out.
- **Per-document deletion is not implemented.** The metadata to support it is
  stored, but there is no endpoint to remove or replace a single document.
- **The keyword index is rebuilt from scratch on every ingest.** This
  re-tokenises the whole corpus, which is negligible at a few hundred chunks
  but would need incremental updates at scale.

## Layout

| File | Purpose |
| --- | --- |
| `chunking.py` | Splits text into overlapping word-based chunks |
| `embeddings.py` | Wraps the OpenAI embeddings API (single and batch) |
| `keyword_search.py` | Tokenisation, BM25 index building and keyword search |
| `similarity.py` | Hand-written cosine similarity — superseded by Chroma, kept for reference |
| `config.py` | API clients, model names, chunking and retrieval settings, Chroma settings |
| `rag.py` | Collection management, vector and hybrid retrieval, rank fusion, prompt assembly, generation |
| `main.py` | FastAPI application — the service itself |
| `ask.py` | Script entry point for exercising the pipeline without HTTP |
| `evaluate.py` | Retrieval evaluation — hit rate and MRR against the eval set, configurable by flags |
| `eval_set.json` | 24 hand-written questions with expected-content anchors |
| `eval_results.jsonl` | One record per evaluation run: configuration, metrics, failures |
| `data/` | Sample corpus (~41,000 words across five documents) |

`rag.py` imports no web framework, and `main.py` contains no retrieval logic.
That separation is why the same pipeline can be driven by a server, a script, or
an evaluation harness without duplicating anything that matters — why swapping
the storage layer for Chroma changed two function bodies rather than the whole
codebase, and why hybrid retrieval could be measured in `evaluate.py` before it
was switched on in the API.

## Setup

```bash
pip install -r requirements.txt
```

Keyword search needs two NLTK data packages, downloaded once:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

Set `OPENAI_API_KEY` in a `.env` file in the project root (see `.env.example`),
then start the server:

```bash
fastapi dev main.py
```

The server indexes everything in `data/` at startup, writing the Chroma
collection to `chroma_db/` (configurable with `CHROMA_PATH`) and building the
keyword index in memory.

To run a single question through the pipeline without starting a server:

```bash
python ask.py
```

To measure retrieval quality against the evaluation set:

```bash
python evaluate.py
```

Flags select the configuration — `--mode vector|hybrid`, `--k`, `--chunk-size`,
`--overlap` — and default to the values in `config.py`.
