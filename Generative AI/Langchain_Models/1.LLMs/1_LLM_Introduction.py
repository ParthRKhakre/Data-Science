from langchain_openai import OpenAI
from dotenv import load_dotenv       
# dotenv is used to load the secret key from environment 

load_dotenv()


llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("What is the capital of india")

print(result)
