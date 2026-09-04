from embeddings import get_embedding, get_embeddings
from chunking import chunk_text
from config import client, chroma_client, COLLECTION_NAME, EMBEDDED_MODEL, LLM


def get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def reset_collection():
    if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(COLLECTION_NAME)
    return get_collection()


def build_index(text, source, chunk_size, overlap):
    chunks = chunk_text(text, chunk_size, overlap)
    embeddings = get_embeddings(chunks, EMBEDDED_MODEL)

    collection = get_collection()
    collection.add(
        ids=[f"{source}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": source} for _ in chunks],
    )
    return collection


def retrieve(query, collection, k: int = 3):
    query_embedding = get_embedding(query, EMBEDDED_MODEL)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    retrieved_chunks = []
    for chunk_id, distance, text, metadata in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        retrieved_chunks.append(
            {
                "score": 1 - distance,
                "id": chunk_id,
                "text": text,
                "source": metadata["source"],
            }
        )

    return retrieved_chunks


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(
        f'<source id="{n}" document="{result["source"]}">\n{result["text"]}\n</source>'
        for n, result in enumerate(retrieved_chunks, start=1)
    )

    prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": """
                     You answer questions using only the sources provided in the user message.
                    The context contains numbered sources wrapped in <source> tags.
                    Each source carries the document attribute naming the file it came from.
                Rules:
                    - Use only information found in the sources. Do not use prior knowledge, and do not infer or speculate beyond what is written.
                    - If the sources do not contain the answer, reply exactly: "I don't know based on the provided context."
                    - If the sources only partially answer the question, say what they support and state clearly what is missing.
                    - Cite the source ids you used, like [1] or [1][3], after the claims they support.
                    - Name the document where the answer came from, especially if the answer comes from more than one source.
                    - Answer concisely.
                        """,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"""
            Context:
            {context}\n\n
            Question:
            {question}
        """,
                }
            ],
        },
    ]
    response = client.responses.create(model=LLM, input=prompt, temperature=0)
    return response.output_text


def answer_question(question, collection):
    retrieved_chunks = retrieve(question, collection)
    answer = generate_answer(question, retrieved_chunks)
    return {"answer": answer, "sources": retrieved_chunks}
