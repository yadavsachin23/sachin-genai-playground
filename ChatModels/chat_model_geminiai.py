from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenAI(model="gemini-1.5-pro", temperature=0.9)
result = model.invoke("Give me a joke about programming.")
print(result.content)
