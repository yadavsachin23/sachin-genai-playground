from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Delhi is the capital of India. It is known for its rich history, culture, and government institutions.",
    "Mumbai is the financial capital of India. It is famous for Bollywood, the stock market, and marine drive.",
    "Bangalore is called the Silicon Valley of India. It is a major hub for technology companies and startups.",
    "Chennai is a coastal city in southern India. It is known for its automobile industry and classical music culture.",
    "Kolkata is known as the cultural capital of India. It is famous for literature, art, and historic landmarks.",
]

query = "Which city is known for its technology industry and startups?"

doc_embedding = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarity_score = cosine_similarity([query_embedding], doc_embedding)[0].tolist()

index, score = sorted(list(enumerate(similarity_score)), key=lambda x: x[1])[-1]
print(f"Most similar document: '{documents[index]}' with similarity score: {score:.4f}")
