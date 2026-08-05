def chunk_text(text, chunk_size, overlap):
    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks
