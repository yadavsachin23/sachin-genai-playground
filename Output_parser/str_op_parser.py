from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="cc",
    task="text-generation",
    huggingfacehub_api_token=token,
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

template1 = PromptTemplate(
    template="Write a detailed note on {topic}?",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Write 5 line summary of {text} in points?",
    input_variables=["text"],
)
# ================== SIMPLE CHAIN ==============================

# chain1 = template1 | model
# result1 = chain1.invoke({"topic": "Artificial Intelligence"})

# chain2 = template2 | model
# result2 = chain2.invoke({"text": result1.content})

# print("\nKey Points:\n", result2.content)

# ================== USING SINGLE CHAIN WITH MULTIPLE PROMPTS ==============================

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "Black Hole"})
print("\nKey Points:\n", result)
