# ── evaluator_node.py ────────────────────────────────────────────────────────

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os, json

from src.schemas import EvalScores, EvalResult, Question_Schema

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
PASS_THRESHOLD = float(os.getenv("PASS_THRESHOLD", 7.0))   # weighted score out of 10

EVALUATOR_SYSTEM = """
You are a strict psychometric evaluator for a Computer Adaptive Test item bank.
You will receive a generated question and score it across six dimensions.

SCORING DIMENSIONS (each 1–10):

1. question_clarity      — Is the stem grammatically clear, unambiguous, and free
                           of multiple interpretations?
2. answer_correctness    — Is the marked solution factually and unambiguously correct?
                           Score 1 if the solution is wrong.
3. distractor_quality    — Are the three wrong options each plausible to a student
                           with partial knowledge? Penalise obviously wrong options.
4. difficulty_match      — Does the question's actual cognitive demand match the
                           requested difficulty (provided below)? ±0.1 tolerance.
5. explanation_quality   — Does the explanation correctly justify the solution and
                           clarify why the distractors are wrong?
6. tag_accuracy          — Do subject / topic / sub_topic correctly describe the
                           question content?

FEEDBACK RULES:
  - Only populate a feedback field when that dimension scores ≤ 7
  - Feedback must be one concrete, actionable sentence (what to fix, not just what's wrong)
  - Do not praise; only report problems

WEIGHTED SCORE (you must compute this):
  answer_correctness  × 0.30
  distractor_quality  × 0.25
  question_clarity    × 0.20
  difficulty_match    × 0.15
  explanation_quality × 0.07
  tag_accuracy        × 0.03

passed = true if weighted_score >= {pass_threshold}

Return ONLY a valid JSON object matching the EvalResult schema. No preamble.
"""

EVALUATOR_HUMAN = """
Requested difficulty : {difficulty}  ({difficulty_label})

--- QUESTION ---
{question_json}
"""

_eval_prompt = ChatPromptTemplate.from_messages([
    ("system", EVALUATOR_SYSTEM),
    ("human",  EVALUATOR_HUMAN),
])

_eval_llm   = ChatGroq(model=GROQ_MODEL, temperature=0.0, api_key=GROQ_API_KEY)
_e_llm      = _eval_llm.with_structured_output(EvalResult)

_difficulty_label = lambda d: (
    "recall / definition"        if d <= 0.3 else
    "conceptual / single-step"   if d <= 0.6 else
    "multi-step reasoning"       if d <= 0.8 else
    "edge case / deep trap"
)

def _feedback_to_prompt(feedback) -> str:
    """Flatten EvalFeedback into a bulleted string for the generator."""
    lines = []
    for field, value in feedback.model_dump().items():
        if value:
            lines.append(f"  • {field}: {value}")
    return "\n".join(lines) if lines else "None."

def evaluate_question_node(state: dict) -> dict:
    """
    Reads from state : Q, Difficulty
    Writes to state  : EvalResult, EvalFeedback (str), RetryCount
    """
    question: Question_Schema = state["Q"]
    difficulty = state.get("Difficulty", 0.5)

    result: EvalResult = (_eval_prompt | _e_llm).invoke({
        "pass_threshold":   PASS_THRESHOLD,
        "difficulty":       difficulty,
        "difficulty_label": _difficulty_label(difficulty),
        "question_json":    json.dumps(question.model_dump(), indent=2),
    })

    return {
        "EvalResult":   result,
        "EvalFeedback": _feedback_to_prompt(result.feedback),
    }