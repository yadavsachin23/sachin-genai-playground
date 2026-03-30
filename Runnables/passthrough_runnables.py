from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
)
from dotenv import load_dotenv
import os

load_dotenv()
openAIToken = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openAIToken)
parser = StrOutputParser()

template = PromptTemplate(
    template="Generate simple and short notes from the following: \n {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Give me 5 short questions on: \n {text}",
    input_variables=["text"],
)

chain = RunnableSequence(
    template | model | parser,
)

pareallel_chain = RunnableParallel(
    {
        "notes": RunnablePassthrough(),
        "questions": RunnableSequence(template2 | model | parser),
    }
)

final_chain = RunnableSequence(
    chain,
    pareallel_chain,
)

text = "AI and humans working together"
result = final_chain.invoke({"topic": text})
print(result)
