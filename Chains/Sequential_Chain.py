from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

prompt1 = PromptTemplate(
    template="Generate a detailed report on {country}?", input_variables=["country"]
)

prompt2 = PromptTemplate(
    template="Give me 5 pointer major summary {text}?",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"country": "India"})
print(result)

chain.get_graph().print_ascii()