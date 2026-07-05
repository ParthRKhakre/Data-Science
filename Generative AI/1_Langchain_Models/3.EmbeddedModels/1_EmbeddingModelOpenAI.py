from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding  = OpenAIEmbeddings(model = "text-embedding-3-large",dimensions="32")
# dimension define the size of vector that capture contextual meaning of the text
# larger the vector more the capture content

result = embedding.embed_query("Delhi is Capital of India")
print(str(result))

