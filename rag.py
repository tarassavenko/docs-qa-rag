from similarity import cosine_similarity
from embeddings import get_embedding,get_embeddings
from chunking import chunk_text
from embeddings import client

def build_index(text,chunk_size,overlap):
    chunks=chunk_text(text,chunk_size,overlap)
    embedded_chunks=get_embeddings(chunks,"text-embedding-3-small")
    return chunks,embedded_chunks


def retrieve(query,embedded_chunks,chunks,k:int=3):
    query_embedding=get_embedding(query,"text-embedding-3-small")
    retrieved_chunks=[]
    for i,emb in enumerate(embedded_chunks):
        score=cosine_similarity(query_embedding,emb)
        retrieved_chunks.append({
            "score":score,
            "id":i,
            "text":chunks[i]
            })
    retrieved_chunks=sorted(retrieved_chunks,key=lambda x:x["score"],reverse=True)
    return retrieved_chunks[:k]

 


