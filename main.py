from rag import build_index,retrieve

with open("data/space.txt",encoding="utf-8") as f:
    text=f.read()
   
chunks,embedded_chunks=build_index(text,150,20)
retrieved_chunks=retrieve("Whats is the outer space",embedded_chunks,chunks)