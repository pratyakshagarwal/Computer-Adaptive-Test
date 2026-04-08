import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


system_prompt = """
You are a brutally honest performance analyst for a student.

Your job is to:
- Identify weak areas precisely
- Identify strong areas
- Detect patterns in mistakes
- Suggest a focused improvement plan

Do NOT give generic advice.
Be specific and actionable.
"""

human_prompt = """
Student Performance Summary:

Overall Accuracy: {accuracy}
Recent Accuracy (last 5): {recent_accuracy}

Weak Topics:
{weak_topics}

Strong Topics:
{strong_topics}

Topic Breakdown:
{topic_stats}

Current Ability (theta): {theta}

Generate:
1. Key weaknesses (max 3)
2. Key strengths (max 3)
3. Mistake patterns
4. A precise improvement plan (step-by-step)
"""

# 3. Create the Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])


I_LLM = ChatGroq(model=GROQ_MODEL, temperature=0.7, api_key=GROQ_API_KEY)

def generate_plan(state):

    chain = prompt_template | I_LLM

    response = chain.invoke({
        "accuracy": state['accuracy'],
        "recent_accuracy": state['recent_accuracy'],
        "weak_topics": state['weak_topics'],
        "strong_topics": state['strong_topics'],
        "topic_stats": state['strong_topics'],
        "theta": state['theta']
    })

    return response