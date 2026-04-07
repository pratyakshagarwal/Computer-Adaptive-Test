# ── generator_node ────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.schemas import Question_Schema

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ── Thresholds (weights must sum to 1.0) ────────────────────────────────────
DIMENSION_WEIGHTS = {
    "answer_correctness":   0.30,
    "distractor_quality":   0.25,
    "question_clarity":     0.20,
    "difficulty_match":     0.15,
    "explanation_quality":  0.07,
    "tag_accuracy":         0.03,
}

# ── System prompt ────────────────────────────────────────────────────────────
GENERATOR_SYSTEM = """
You are an expert {exam} question author.

Your task is to produce ONE multiple-choice question that will be used in a
Computer Adaptive Test. The question must be unambiguous, pedagogically sound,
and match the requested difficulty exactly.

DIFFICULTY SCALE (target value supplied in the request):
  0.1–0.3  → recall / basic definition, one concept, no calculations
  0.4–0.6  → conceptual understanding, single-step reasoning or small calculation
  0.7–0.8  → multi-step reasoning, requires connecting two concepts
  0.9–1.0  → edge cases, common misconceptions as distractors, deep understanding

DISTRACTOR RULES (the three wrong options):
  - Each must be plausible to a student who partially understands the topic
  - No option may be obviously absurd or unrelated
  - Only ONE option must be correct — verify this before outputting
  - Avoid options that overlap in meaning or are supersets/subsets of each other

OUTPUT CONTRACT:
  - Return a single JSON object matching the schema exactly
  - "opt"        → keys A B C D, values are the option texts
  - "solution"   → one of A B C D
  - "difficulty" → float matching the requested value ±0.05
  - "tags"       → {{"subject": ..., "topic": ..., "sub_topic": ...}}
  - No extra fields, no markdown, no preamble
"""

# ── Human prompt ─────────────────────────────────────────────────────────────
GENERATOR_HUMAN = """
Exam         : {exam}
Subject(s)   : {subjects}
Topic(s)     : {topics}
Difficulty   : {difficulty}  ({difficulty_label})

--- HISTORY (avoid repeating these sub_topics) ---
{history}

--- EVALUATOR FEEDBACK (from previous attempt, address all points) ---
{feedback}

Generate a question now.
"""

_difficulty_label = lambda d: (
    "recall / definition"        if d <= 0.3 else
    "conceptual / single-step"   if d <= 0.6 else
    "multi-step reasoning"       if d <= 0.8 else
    "edge case / deep trap"
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATOR_SYSTEM),
    ("human",  GENERATOR_HUMAN),
])

_llm   = ChatGroq(model=GROQ_MODEL, temperature=0.3, api_key=GROQ_API_KEY)
_q_llm = _llm.with_structured_output(Question_Schema)

def generate_question_node(state: dict) -> dict:
    """
    Reads from state:
        Subjects, Topics, Exam, Difficulty, History, EvalFeedback, RetryCount
    Writes to state:
        Q, RetryCount
    """
    difficulty = state.get("Difficulty", 0.5)
    feedback   = state.get("EvalFeedback") or "None — first attempt."

    response = (_prompt | _q_llm).invoke({
        "exam":              state.get("Exam") or "General Practice",
        "subjects":          ", ".join(state.get("Subjects", [])),
        "topics":            ", ".join(state.get("Topics", [])),
        "difficulty":        difficulty,
        "difficulty_label":  _difficulty_label(difficulty),
        "history":           state.get("History") or "No history yet.",
        "feedback":          feedback,
    })

    return {
        "Q":          response,
        "RetryCount": state.get("RetryCount", 0),   # reset only by router
    }
