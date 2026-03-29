from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

chat_history = []

file_path = os.path.join(os.path.dirname(__file__), "history.txt")
with open(file_path) as f:
    chat_history.extend(f.readlines())

# print(chat_history)

result = chat_template.invoke(
    {"query": "Where is my refund?", "chat_history": chat_history}
)

print(result, "====================")
