from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = "",dimension = 32)

documents = [
    "USA is Rogue Nation",
    "Delhi is capital of india",
    "Kolkata is capital of West Bengal",
    "Noida is in UP"    
]

result = embedding.embed_documents(documents)

print(str(result))
