from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()

os.environ["HF_HOME"] = "D:/huggingface_cache"

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print("Token loaded:", token[:10] if token else "NOT FOUND")  # ✅ no indent

login(token=token)

hf_wrapper = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        temperature=0.7,
    ),
)

model = ChatHuggingFace(llm=hf_wrapper)
result = model.invoke("What is the capital of India?")
print(result.content)
