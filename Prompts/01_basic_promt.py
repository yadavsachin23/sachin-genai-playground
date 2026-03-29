from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model="gpt-4", temperature=0.7)
result = chat.invoke("Write 10 on India")
print(result.content)
