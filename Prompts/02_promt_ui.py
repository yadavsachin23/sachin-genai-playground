from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import json
import os

# Load environment variables
load_dotenv()

# Initialize the model
chat = ChatOpenAI(model="gpt-4", temperature=0.7)

# Load the prompt template
json_path = os.path.join(os.path.dirname(__file__), "research_summary_template.json")
with open(json_path, "r") as f:
    data = json.load(f)
    template = PromptTemplate(
        template=data["template"],
        input_variables=data["input_variables"],
    )

# ── UI ────────────────────────────────────────────────────────────────────────

st.header("Research Assistant")

paper_input = st.selectbox(
    "📄 Select a research paper",
    (
        "Attention Is All You Need (Vaswani et al., 2017)",
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "GPT-4 Technical Report (OpenAI, 2023)",
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020)",
    ),
)

style_input = st.radio(
    "✍️ Writing Style",
    options=["Formal", "Informal", "Technical"],
    horizontal=True,
    captions=["Academic tone", "Conversational tone", "Jargon-heavy"],
)

length_input = st.select_slider(
    "📏 Response Length",
    options=["Short", "Medium", "Long"],
    value="Medium",
)

# ── Generation ────────────────────────────────────────────────────────────────

if st.button("Generate Response"):
    chain = template | chat
    result = chain.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input,
        }
    )
    st.write(result.content)
