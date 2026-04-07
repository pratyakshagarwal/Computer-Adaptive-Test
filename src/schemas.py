from pydantic import BaseModel, Field
from typing import TypedDict, Optional, List, Dict


class Options(BaseModel): A: str; B: str; C: str; D: str
class Tags(BaseModel): subject: str; topic: str; sub_topic: str

class Question(BaseModel):
    q: str
    opt: dict
    solution: str
    explanation: str 
    difficulty: float = Field(ge=0.1, le=1)
    tags: dict

class UserSession(TypedDict):
    Subjects: list[str]   # User prefered subjects 
    Topcis: list[str]     # Topics to ask questions on 

    Exam: Optional[str | None]   # The Exam user is preparing for

    Difficulty: float = Field(ge=0.1, le=1)      # The current difficulty
    Q: Question          # Questions Based on current difficulty

    History: List[Question]   # History of all the questions asked
    Evaluator_Feedback =  List[str]

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


class EvalScores(BaseModel):
    question_clarity: int = Field(ge=1, le=10)
    answer_correctness: int = Field(ge=1, le=10)
    distractor_quality: int = Field(ge=1, le=10)   # plausibility of wrong opts
    difficulty_match: int = Field(ge=1, le=10)
    explanation_quality: int = Field(ge=1, le=10)
    tag_accuracy: int = Field(ge=1, le=10)

class EvalFeedback(BaseModel):
    question_clarity: str | None = None
    answer_correctness: str | None = None
    distractor_quality: str | None = None
    difficulty_match: str | None = None
    explanation_quality: str | None = None
    tag_accuracy: str | None = None

class EvalResult(BaseModel):
    scores: EvalScores
    feedback: EvalFeedback    
    passed: bool
    weighted_score: float



# -------------------------
# Schemas
# -------------------------
class SessionCreate(BaseModel):
    subjects: List[str] = Field(min_length=1)
    topics: List[str] = Field(min_length=1)
    exam: Optional[str] = None


class GenerateQuestionRequest(BaseModel): session_id: str
class GenerateInsightRequest(BaseModel): session_id: str


class QuestionResponse(BaseModel):
    id: str
    q_text: str
    options: Dict[str, str]
    solution: str
    explanation: str
    difficulty: float
    subject: Optional[str]
    topic: Optional[str]
    sub_topic: Optional[str]

    class Config:
        from_attributes = True


class SubmitAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    user_answer: str