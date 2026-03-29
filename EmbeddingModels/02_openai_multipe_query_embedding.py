from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=30)

documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Bangalore is the IT hub of India.",
]

result = embeddings.embed_documents(documents)
print(str(result))
