from pydantic import BaseModel


class AnswerSubmission(BaseModel):
    session_id: str
    question_id: str
    answer: str