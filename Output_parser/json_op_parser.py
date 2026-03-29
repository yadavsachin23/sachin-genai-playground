from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template1 = PromptTemplate(
    template="Give me 5 fact about {topic} \n {format_inst}",
    input_variables=["topic"],
    partial_variables={"format_inst": parser.get_format_instructions()},
)
chain = template1 | model | parser
result = chain.invoke({"topic": "Gym"})

# prompt = template1.format()
# result = model.invoke(prompt)
# print(result)
# parsed_result = parser.parse(result.content)
print(result)
