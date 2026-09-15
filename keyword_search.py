import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

STEMMER = PorterStemmer()
VALID_TOKEN = re.compile(r"^\w+(?:[.,]\w+)*$")
STOPWORDS = set(stopwords.words("english"))


def tokenise(text, remove_stopwords=True, stem=True):
    tokens = word_tokenize(text.casefold())

    results = []
    for token in tokens:
        if not VALID_TOKEN.match(token):
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        if stem:
            token = STEMMER.stem(token)
        results.append(token)

    return results


def build_keyword_index(collection):
    records = collection.get()
    if not records["documents"]:
        return None
    tokenised_docs = [tokenise(document) for document in records["documents"]]
    return {
        "bm25": BM25Okapi(tokenised_docs),
        "ids": records["ids"],
        "documents": records["documents"],
        "metadatas": records["metadatas"],
    }


def search_keywords(keyword_index, query, n):
    query_tokens = tokenise(query)
    if not query_tokens:
        return []

    scores = keyword_index["bm25"].get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

    results = []
    for i, score in ranked[:n]:
        if score <= 0:
            break
        results.append(
            {
                "score": float(score),
                "id": keyword_index["ids"][i],
                "text": keyword_index["documents"][i],
                "source": keyword_index["metadatas"][i]["source"],
            }
        )
    return results
