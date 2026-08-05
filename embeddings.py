from config import client


def get_embedding(text, model):
    text = text.replace("\n", " ")
    return (
        client.embeddings.create(model=model, input=text, encoding_format="float")
        .data[0]
        .embedding
    )


def get_embeddings(texts, model):
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return [item.embedding for item in response.data]
