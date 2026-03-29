from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize the model
chat = ChatOpenAI(model="gpt-4", temperature=0.7)

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🤖 AI Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Type your message here...")

if user_input:
    # Display and store user message
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chat.invoke(st.session_state.chat_history)
            st.write(result.content)

    st.session_state.messages.append({"role": "assistant", "content": result.content})
    st.session_state.chat_history.append(
        {"role": "assistant", "content": result.content}
    )

# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# chat_history = []

# while True:
#     userInput = input("You: ")
#     chat_history.append(userInput)
#     if userInput.lower() in ["exit", "quit"]:
#         print("Exiting the chatbot. Goodbye!")
#         break
#     chat = ChatOpenAI(model="gpt-4", temperature=0.7)
#     result = chat.invoke(chat_history)
#     chat_history.append(result.content)
#     print("Chatbot:", result.content)

# print("Chatbot session ended.", chat_history)
