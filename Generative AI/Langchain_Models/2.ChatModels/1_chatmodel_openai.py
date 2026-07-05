from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=2,max_completion_tokens=200)

# temperature is parameter that  controls the randomness of language models output
# It affects how creative or deterministic the responses are 
# Lower values : 0.0 to 0.3 
# Higher Values : 0.7 to 1.5
# max_completion_tokens means what maximum tokens AI can use

result = model.invoke("What is the capital of india")
print(result.content)

