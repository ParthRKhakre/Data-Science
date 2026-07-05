from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
# hugging face pipeline is used for using local LLMs
import os 

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    pipeline_kwargs= dict(
        temperature = 0.5, 
        max_new_token = 100
    )
)

model = ChatHuggingFace(llm)
result = model.invoke("What is Capital of India")
print(result.content)
