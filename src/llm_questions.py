import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


class Question_Schema(BaseModel):
    """Schema for a multiple-choice question with metadata."""
    q: str = Field(description="The actual question text")
    opt: Dict[str, str] = Field(
        description="A dictionary of options where keys are labels (e.g., 'A', 'B') and values are the option text"
    )
    solution: str = Field(description="The correct option key (e.g., 'A')")
    explanation: str = Field(description="A detailed explanation of why the solution is correct")
    difficulty: float = Field(description="Difficulty level from 0.1 (easy) to 1.0 (hard)")
    tags: Dict[str, str] = Field(
        description="The specific topics or categories the question is derived from (e.g., {'subject': 'Physics', 'topic': 'Kinematics'})"
    )

# 1. Define the System Prompt to set the "Persona" and "Rules"
system_prompt = """
You are an expert educational content generator.

You MUST generate a valid multiple-choice question.

STRICT REQUIREMENTS:
- Output must EXACTLY match the required JSON schema
- Do not add extra fields
- Do not nest fields incorrectly
- "opt" must contain ONLY A, B, C, D
- "solution" must be one of A, B, C, D
- "difficulty" must be between 0.1 and 1.0
    Difficulty Guidelines:
    - 0.2 → basic definitions
    - 0.5 → conceptual + small reasoning
    - 0.8 → multi-step reasoning or tricky distractors
    - 1.0 → edge cases / traps / deep understanding
- "tags" must include subject, topic, sub_topic
- Do NOT repeat previous questions or sub_topics
"""
# 2. Define the Human Prompt using variables from your UserSession
human_prompt = """
Generate a question using:

Subjects: {subjects}
Topics: {topics}
Exam: {exam}
Difficulty: {difficulty}

Avoid Topics with High frequnecy and try to distribute it in many subtopicss:
{history}

Return ONLY a valid JSON object matching the schema.
"""

# 3. Create the Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])


LLM = ChatGroq(model=GROQ_MODEL, temperature=0.7, api_key=GROQ_API_KEY)
Q_LLM = LLM.with_structured_output(Question_Schema)

def generate_question_node(state):
    history_summary = state.get("History", {})

    chain = prompt_template | Q_LLM

    response = chain.invoke({
        "subjects": ", ".join(state.get("Subjects", [])),
        "topics": ", ".join(state.get("Topics", [])),  # ✅ FIXED TYPO
        "exam": state.get("Exam") or "General Practice",
        "difficulty": state.get("Difficulty", 0.5),
        "history": history_summary
    })

    return response