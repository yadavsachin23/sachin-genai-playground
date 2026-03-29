from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="Fact_1", description="Fact 1 about topic"),
    ResponseSchema(name="Fact_2", description="Fact 2 about topic"),
    ResponseSchema(name="Fact_3", description="Fact 3 about topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about {topic}\n{format_inst}",
    input_variables=["topic"],
    partial_variables={"format_inst": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"topic": "Black Hole"})

print(result)
print(result["Fact_1"])
print(result["Fact_2"])
print(result["Fact_3"])
