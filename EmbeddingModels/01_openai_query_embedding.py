from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=30)

result = embeddings.embed_query("Delhi is capital of India")
print(str(result))
