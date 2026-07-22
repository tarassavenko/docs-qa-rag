import os
from openai import OpenAI
from dotenv import load_dotenv

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

print(response.output_text)