
from similarity import cosine_similarity
from embeddings import get_embedding,get_embeddings
from chunking import chunk_text



with open("data/space.txt",encoding="utf-8") as f:
    text=f.read()
   


chunks=chunk_text(text,20,0)


query_embedding=get_embedding("Who was the first human to achieve crewed Earth orbit?","text-embedding-3-small")


embedded_chunks=get_embeddings(chunks,"text-embedding-3-small")
scores=[]
for i,emb in enumerate(embedded_chunks):
    score=cosine_similarity(query_embedding,emb)
    scores.append((score,i))

best_score,best_index=max(scores)
print(best_score)
print(f"{chunks[best_index]}\n")
print(chunks[17],chunks[16])
