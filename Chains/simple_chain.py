from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="Tell me a fact about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

# Chain = prompt | model | parser
chain = template | model | parser

result = chain.invoke({"topic": "Black Holes"})
print(result)
