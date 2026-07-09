from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
    )


result = model.invoke("Write a 5 line poem on cricket ?")

# Query input message inside the invoke function is known as prompt 
# Prompt are of two types : 
# Text based prompt
# Multimodel prompt (text + image/file/audio/video,etc)

# LLM heavily depend on the prompt 
# There are a lot ot methods to create prompt  
# 1.Static prompt
# 2.Dynamic Prompt 


