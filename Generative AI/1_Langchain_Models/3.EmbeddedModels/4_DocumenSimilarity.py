from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name =  "LiquidAI/LFM2.5-Embedding-350M"
)

documents = [
    "Artificial Intelligence is transforming industries by automating complex tasks.",
    "Machine learning algorithms improve their performance by learning from data.",
    "The capital of India is New Delhi, which is located in the northern part of the country",
    "Embedding models convert text into numerical vectors that capture semantic meaning."
    "LangChain helps developers build applications powered by large language models."
]

query = "Tell me about Embedding models"

document_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

score = cosine_similarity([query_embedding],document_embedding)[0]

index,score = sorted(list(enumerate(score)),key = lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score",score)


