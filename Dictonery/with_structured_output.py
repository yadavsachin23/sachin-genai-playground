from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()


class Review(TypedDict):

    key_itrem: Annotated[str, "The key argument of the review"]
    summary: Annotated[str, "A concise summary of the review in 2-3 sentences"]
    sentiment: Annotated[
        str,
        "The overall sentiment of the review, categorized as Positive, Negative, or Neutral",
    ]
    pros: Annotated[
        Optional[list[str]],
        "A list of positive aspects mentioned in the review, if any",
    ]
    cons: Annotated[
        Optional[list[str]],
        "A list of negative aspects mentioned in the review, if any",
    ]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """I've been using this research assistant for the past few weeks and it has genuinely transformed the way I approach academic reading. As someone who regularly has to go through dense ML papers, being able to get a crisp, well-structured summary in seconds is invaluable. The writing style selector is a standout feature — switching between Formal and Technical modes gives me exactly the kind of output I need depending on whether I'm preparing a slide deck or doing a deep methodology review. The summaries are well-structured, consistently covering the core problem, approach, key findings, and real-world implications. What impressed me most was how accurately it captured the central argument of the Attention Is All You Need paper in Short mode — three sentences that actually told me something useful, not just filler. The Streamlit interface is clean, intuitive, and loads quickly. Highly recommended.
Pros:

Three distinct writing styles with clear tonal differences between them
Short mode is genuinely concise — not just a truncated long summary
Clean, minimal UI with no clutter or unnecessary fields
Structured output consistently covers problem, method, findings, and impact
Fast response time even for complex papers

Cons:

Only five papers available — would love more options
No option to copy or download the generated summary
All papers are AI/ML focused — limits use for other domains"""
)

print(result)
print(result["summary"])
print(result["sentiment"])
