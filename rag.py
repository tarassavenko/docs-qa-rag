from similarity import cosine_similarity
from embeddings import get_embedding, get_embeddings
from chunking import chunk_text
from config import client, EMBEDDED_MODEL, LLM


def build_index(text, chunk_size, overlap):
    chunks = chunk_text(text, chunk_size, overlap)
    embedded_chunks = get_embeddings(chunks, EMBEDDED_MODEL)
    return chunks, embedded_chunks


def retrieve(query, embedded_chunks, chunks, k: int = 3):
    query_embedding = get_embedding(query, EMBEDDED_MODEL)
    retrieved_chunks = []
    for i, emb in enumerate(embedded_chunks):
        score = cosine_similarity(query_embedding, emb)
        retrieved_chunks.append({"score": score, "id": i, "text": chunks[i]})
    retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True)
    return retrieved_chunks[:k]


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(
        f'<source id="{n}">\n{result["text"]}\n</source>'
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
                Rules:
                    - Use only information found in the sources. Do not use prior knowledge, and do not infer or speculate beyond what is written.
                    - If the sources do not contain the answer, reply exactly: "I don't know based on the provided context."
                    - If the sources only partially answer the question, say what they support and state clearly what is missing.
                    - Cite the source ids you used, like [1] or [1][3], after the claims they support.
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


def answer_question(question, embedded_chunks, chunks):
    retrieved_chunks = retrieve(question, embedded_chunks, chunks)
    answer = generate_answer(question, retrieved_chunks)
    return {"answer": answer, "sources": retrieved_chunks}
