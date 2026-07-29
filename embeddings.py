import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=True)
api_key=os.getenv("OPENAI_API_KEY")

client=OpenAI()
if __name__=="__main__":
 if api_key and api_key.startswith("sk-proj-"):
    print("Api Key was found")
 else:
    print("Error,Api key was not found!")


def get_embedding(text,model):
    text=text.replace("\n"," ")
    return client.embeddings.create(model=model,input=text,encoding_format="float").data[0].embedding

def get_embeddings(texts,model):
   texts=[t.replace("\n"," ") for t in texts]
   response=client.embeddings.create(model=model,input=texts,encoding_format="float")
   return [item.embedding for item in response.data]