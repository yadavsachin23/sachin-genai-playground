from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch
import os

load_dotenv()
openAIToken = os.getenv("OPENAI_API_KEY")
anthropicToken = os.getenv("CLAUDEAI_API_KEY")

openAIModel = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openAIToken)
anthropicModel = ChatAnthropic(
    model="claude-sonnet-4-6", anthropic_api_key=anthropicToken
)

parser = StrOutputParser()


class FeedbackChain(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback, either 'positive' or 'negative'."
    )


parser2 = PydanticOutputParser(pydantic_object=FeedbackChain)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the given feedback text into positive and negative \n {topic}"
    "{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{topic}",
    input_variables=["topic"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{topic}",
    input_variables=["topic"],
)

classify_chain = prompt1 | openAIModel | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | openAIModel | parser),
    (lambda x: x.sentiment == "negative", prompt3 | openAIModel | parser),
    RunnableLambda(lambda x: "Invalid sentiment"),
)

chain = classify_chain | branch_chain

POSITIVE_FEEDBACK = "I had a great experience with your product! The customer service was excellent and the quality exceeded my expectations."
NEGATIVE_FEEDBACK = "I had a terrible experience with your product. The customer service was awful and the quality was very poor."

print(chain.invoke({"topic": POSITIVE_FEEDBACK}))
