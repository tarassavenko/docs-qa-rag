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

doc="""In the summer of 2006,Italy won the Mundial of 2006.
       They beat France in the penalties 5-4.
       That Year,Canavaro was the best football player of the world.
       """

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

response1=get_embedding(doc,"text-embedding-3-small")
response2=get_embedding("Did Italy won Mundial on 2006?","text-embedding-3-small")

score=cosine_similarity(response1,response2)
print(score)
     

