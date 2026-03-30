from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()
openAIToken = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openAIToken)
parser = StrOutputParser()

template = PromptTemplate(
    template="Generate a tweet about \n{topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Generate a LinkedIn post about \n{topic}",
    input_variables=["topic"],
)

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(template | model | parser),
        "linkedin": RunnableSequence(template2 | model | parser),
    }
)

result = parallel_chain.invoke({"topic": "AI"})
print(result["tweet"])
print(result["linkedin"])
