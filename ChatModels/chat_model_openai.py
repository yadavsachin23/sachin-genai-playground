from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.9)
result = model.invoke("Give me a joke about programming.")
print(result.content)
