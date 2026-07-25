import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv(override=True)
api_key=os.getenv("OPENAI_API_KEY")

if api_key and api_key.startswith("sk-proj-"):
    print("Api Key was found")
else:
    print("Error,Api key was not found!")


client=OpenAI()
response=client.responses.create(
    model="gpt-4o-mini",
    input="Whats is an embedding?"


)

#print(response.output_text)


doc1="""The Greek Cuisine is considered amongst the best in the world.
        One of the best dishes of the Greek Cuisine is Mousakas.
        It was ranked #1 in 2024.
        """

doc="""In the summer of 2006,Italy won the Mundial of 2006.
       They beat France in the penalties 5-4.
       That Year,Canavaro was the best football player of the world.
       """
doc2="""One genre of movies is science fiction.
        Lord of the Rings in considered a science fiction movie but it is also considered the best trilogy in cinema history.
        The duration of the whokle trilogy is estimated around 12 hours
        """
documents=[doc,doc1,doc2]
embeddings_dict={}  

def get_embedding(text,model):
    text=text.replace("\n"," ")
    return client.embeddings.create(model=model,input=text,encoding_format="float").data[0].embedding

def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)

    product_dot=np.dot(a,b) #To eswteriko ginomeno twn 2 arrays

    norm_a=np.linalg.norm(a) #To mhkos tou dianusmatos a
    norm_b=np.linalg.norm(b) #To mhkos tou dianusmatos b

    return product_dot/(norm_a * norm_b)

for i,doc in enumerate(documents):
    snippet=" ".join(doc.split()[:10])

    embeddings_dict[f"doc{i+1}"]={
        "snippet":snippet,
        "embedding":get_embedding(doc,"text-embedding-3-small")
    }

question_embedding=get_embedding("Is Greek Cuisine the best in the world?And if yes which year?","text-embedding-3-small")

similarities={}

for doc_name,doc_data in embeddings_dict.items():
    similarity=cosine_similarity(question_embedding,doc_data["embedding"])
    print(doc_name,similarity)

    similarities[doc_name]=similarity

print(similarities)    

best_doc=max(similarities,key=similarities.get)
print(best_doc,similarities[best_doc],embeddings_dict[best_doc]["snippet"])

