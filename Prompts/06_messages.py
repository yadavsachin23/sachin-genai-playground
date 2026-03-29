from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
chat = ChatOpenAI()

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
]

result = chat.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages.content)
