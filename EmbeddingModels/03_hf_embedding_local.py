from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Input text
text = "Delhi is the capital of India."
documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Bangalore is the IT hub of India.",
]

# Generate embedding
# result = model.embed_query(text)
multiple_result = model.embed_documents(documents)

# Print result
# print(result)
print(multiple_result)
