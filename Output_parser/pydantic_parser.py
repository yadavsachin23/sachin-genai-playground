from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
)
model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(gt=18, description="The person's age")
    city: str = Field(description="The city where the person lives")


output_parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""You are a data generator. 
Your task is to generate details of a fictional {country} person.

STRICT RULES:
- Return ONLY a valid JSON object
- Do NOT write any code
- Do NOT add any explanation
- Do NOT add anything before or after the JSON

{format_instructions}
""",
    input_variables=["country"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

Chain = template | model | output_parser
result = Chain.invoke({"country": "Indian"})

print(result)
print(result.name)
print(result.age)
print(result.city)

# ================= WORKING ===================

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# import os

# load_dotenv()

# token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation",
#     huggingfacehub_api_token=token,
#     max_new_tokens=200,
#     do_sample=False,
# )

# model = ChatHuggingFace(llm=llm)


# class Person(BaseModel):
#     name: str = Field(description="The person's name")
#     age: int = Field(gt=18, description="The person's age")
#     city: str = Field(description="The city where the person lives")


# output_parser = PydanticOutputParser(pydantic_object=Person)

# template = PromptTemplate(
#     template=(
#         "Generate details of one fictional person from {country}.\n"
#         "Return ONLY valid JSON.\n"
#         "Do not add explanation, markdown, code fences, or extra text.\n"
#         "{format_instructions}"
#     ),
#     input_variables=["country"],
#     partial_variables={"format_instructions": output_parser.get_format_instructions()},
# )

# chain = template | model | output_parser

# try:
#     result = chain.invoke({"country": "India"})
#     print(result)
#     print(result.name)
#     print(result.age)
#     print(result.city)
# except Exception as e:
#     print("Parsing failed:", e)
