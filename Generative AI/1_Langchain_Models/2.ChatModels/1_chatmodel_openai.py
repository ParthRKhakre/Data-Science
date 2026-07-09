from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=2,max_completion_tokens=200)

# temperature is parameter that  controls the randomness of language models output
# It affects how creative or deterministic the responses are 
# Lower values : 0.0 to 0.3 
# Higher Values : 0.7 to 1.5
# max_completion_tokens means what maximum tokens AI can use

# Temperature if kept = 0 then you will get the same output every time 
# as we change the value to 0.5 the output from llm changes a little 
# and soon the output generated is creative.
# Hence when you want you application to respond the same manner for a 
# defined input then keep temperature  = 0

result = model.invoke("What is the capital of india")
print(result.content)

