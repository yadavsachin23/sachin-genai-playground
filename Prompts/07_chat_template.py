from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful {domain} assistant."),
        ("human", "Explain the concept of {concept}."),
        # SystemMessage(content="You are a helpful {domain} assistant."),
        # HumanMessage(content="Explain the concept of {concept}."),
    ]
)

prompt = chat_template.format_prompt(domain="cricket", concept="pitch")
print(prompt)
