from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()
openAIToken = os.getenv("OPENAI_API_KEY")
anthropicToken = os.getenv("CLAUDEAI_API_KEY")

openAIModel = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openAIToken)
anthropicModel = ChatAnthropic(
    model="claude-sonnet-4-6", anthropic_api_key=anthropicToken
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate simple and short notes from the following: \n {topic}",
    input_variables=["topic"],
)

# ✅ Fix 1 - changed {text} to {topic}
prompt2 = PromptTemplate(
    template="Give me 5 short questions on: \n {text}",
    input_variables=["text"],
)

# ✅ Fix 2 - changed {text} to {notes} and {questions}
prompt3 = PromptTemplate(
    template="Merge the following provided notes and questions into a single document.\nNotes -> {notes}\nQuestions -> {questions}",
    input_variables=["notes", "questions"],
)

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | openAIModel | parser,
        "questions": prompt2 | anthropicModel | parser,
    }
)

merge_chain = prompt3 | openAIModel | parser

chain = parallel_chain | merge_chain


text = """Here's a 5-pointer major summary of "India: A Comprehensive Report":

1. **Location and Geography**: India is a federal parliamentary republic located in South Asia, the seventh-largest country by land area, and the second-most populous country with a diverse geography, including mountains, deserts, forests, and coastal regions.

2. **Diverse Climate**: India has a diverse climate, ranging from tropical to alpine, with five climate zones: tropical (southern states), subtropical (eastern states), temperate (northern states), alpine (Himalayan mountain range), and desert (western states).

3. **Mixed Economy**: India has a mixed economy with both public and private sectors playing a significant role, a large service sector (over 60% of GDP), and major exports including textiles, steel, and machinery.      

4. **Diverse Demographics**: India is a young and diverse country with a population of over 1.38 billion people, a population growth rate of 1.2%, and a sex ratio of 944 females per 1,000 males, with over 1,600 languages and dialects spoken across the country.

5. **Growing Economy and Middle Class**: India has a growing middle class with a large number of people earning above the poverty line, and the country is a major exporter of goods to partners including the United States, China, and the European Union."""

result = chain.invoke({"topic": text, "text": text})
print(result)

chain.get_graph().print_ascii()
