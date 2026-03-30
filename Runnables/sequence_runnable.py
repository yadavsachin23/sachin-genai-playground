from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()
openAIToken = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openAIToken)
parser = StrOutputParser()

template = PromptTemplate(
    template="Give me a joke \n{topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Explain me that joke \n{text}",
    input_variables=["text"],
)

chain = RunnableSequence(template, model, parser, template2, model, parser)

topic = "AI and humans working together"
result = chain.invoke({"topic": topic})
print(result)
