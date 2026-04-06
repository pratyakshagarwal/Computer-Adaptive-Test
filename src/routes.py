from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid, copy
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Float, TIMESTAMP, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.dialects.postgresql import JSONB

from src.llm_questions import generate_question_node
from src.llm_insight import generate_plan
from src.irt_lite import IRTEngine

# -------------------------
# DB Setup
# -------------------------
DATABASE_URL = "postgresql://postgres:Sid1002@localhost:5432/postgres"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# -------------------------
# DB Dependency
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# DB Models
# -------------------------
class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)
    user_hash = Column(String, index=True)

    subjects = Column(JSONB)
    topics = Column(JSONB)
    exam = Column(String, nullable=True)

    difficulty = Column(Float, default=0.5)
    theta = Column(Float, default=0.5)
    topic_distribution = Column(JSONB)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)


class QuestionModel(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, index=True)  # critical for queries

    q_text = Column(String)
    options = Column(JSONB)
    solution = Column(String)
    explanation = Column(String)

    difficulty = Column(Float)

    subject = Column(String)
    topic = Column(String)
    sub_topic = Column(String)

    created_at = Column(TIMESTAMP, default=datetime.utcnow)

class AttemptModel(Base):
    __tablename__ = "attempts"

    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    question_id = Column(String, index=True)

    user_answer = Column(String)
    is_correct = Column(Boolean)

    created_at = Column(TIMESTAMP, default=datetime.utcnow)

class ModelInsight(Base):
    __tablename__ = 'insight'

    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    weak_topics = Column(JSONB)
    response = Column(String)

# -------------------------
# Create Tables
# -------------------------
Base.metadata.create_all(bind=engine)

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

# -------------------------
# Router
# -------------------------
router = APIRouter()

# -------------------------
# Endpoints
# -------------------------
@router.post("/start-session")
def start_session(data: SessionCreate, db: Session = Depends(get_db)):
    user_hash = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    new_session = SessionModel(
        id=session_id,
        user_hash=user_hash,
        subjects=data.subjects,
        topics=data.topics,
        exam=data.exam,
        difficulty=0.5,
    )

    db.add(new_session)
    db.commit()

    return {
        "message": "Session created",
        "user_hash": user_hash,
        "session_id": session_id
    }


@router.post("/generate_questions", response_model=QuestionResponse)
def generate_question(data: GenerateQuestionRequest, db: Session = Depends(get_db)):

    session = db.query(SessionModel).filter(SessionModel.id == data.session_id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build clean LLM state
    state = {
        "Subjects": session.subjects,
        "Topics": session.topics,  
        "Exam": session.exam,
        "Difficulty": session.difficulty,
        "History": session.topic_distribution
    }

    try:
        q_state = generate_question_node(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    q_model = QuestionModel(
        id=str(uuid.uuid4()),
        session_id=session.id,

        q_text=q_state.q,
        options=q_state.opt,
        solution=q_state.solution,
        explanation=q_state.explanation,

        difficulty=q_state.difficulty,

        subject=q_state.tags.get("subject"),
        topic=q_state.tags.get("topic"),
        sub_topic=q_state.tags.get("sub_topic"),
    )

    key = (q_model.topic or "") + "_" + (q_model.sub_topic or "")
    dist = copy.copy(session.topic_distribution) or {}
    dist[key] = dist.get(key, 0) + 1
    session.topic_distribution = dist

    db.add(q_model)
    db.commit()
    db.refresh(q_model)

    return q_model


@router.post("/submit-answer")
def submit_answer(data: SubmitAnswerRequest, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(
        SessionModel.id == data.session_id
    ).first()

    if not session:
        raise HTTPException(404, "Session not found")

    question = db.query(QuestionModel).filter(
        QuestionModel.id == data.question_id
    ).first()

    if not question:
        raise HTTPException(404, "Question not found")

    is_correct = data.user_answer == question.solution

    attempt = AttemptModel(
        id=str(uuid.uuid4()),
        session_id=data.session_id,
        question_id=data.question_id,
        user_answer=data.user_answer,
        is_correct=is_correct
    )
    db.add(attempt)

    # initialize engine with CURRENT theta
    irt = IRTEngine(ability=0.5)
    irt.theta = session.theta  # override with stored value

    # update using question difficulty (external scale)
    irt.update(
        correct=is_correct,
        question_difficulty=question.difficulty
    )

    # compute next difficulty
    d_irt = irt.update_difficulty()
    new_diff = irt.from_irt(d_irt)

    session.theta = irt.theta
    session.difficulty = new_diff

    db.commit()

    return {
        "correct": is_correct,
        "correct_answer": question.solution,
        "explanation": question.explanation,
        "new_difficulty": new_diff
    }

@router.get("/get_insights/{session_id}")
def get_insights(session_id: str, db: Session = Depends(get_db)):

    session = db.query(SessionModel).filter(
        SessionModel.id == session_id
    ).first()

    if not session:
        raise HTTPException(404, "Session not found")

    questions = db.query(QuestionModel).filter(
        QuestionModel.session_id == session_id
    ).all()

    attempts = db.query(AttemptModel).filter(
        AttemptModel.session_id == session_id
    ).all()

    if not attempts:
        return {"message": "Not enough data"}

    # -------------------------
    # AGGREGATION
    # -------------------------
    total = len(attempts)
    correct = sum(a.is_correct for a in attempts)
    accuracy = correct / total

    from collections import defaultdict
    topic_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    q_map = {q.id: q for q in questions}

    for a in attempts:
        q = q_map.get(a.question_id)
        if not q:
            continue

        topic = q.topic
        topic_stats[topic]["total"] += 1
        if a.is_correct:
            topic_stats[topic]["correct"] += 1

    weak_topics = [
        t for t, v in topic_stats.items()
        if v["correct"] / v["total"] < 0.5
    ]

    strong_topics = [
        t for t, v in topic_stats.items()
        if v["correct"] / v["total"] > 0.8
    ]

    last_5 = attempts[-5:]
    recent_accuracy = sum(a.is_correct for a in last_5) / len(last_5)

    # -------------------------
    # LLM CALL
    # -------------------------
    state = {
        "accuracy": accuracy,
        "recent_accuracy": recent_accuracy,
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
        "topic_stats": dict(topic_stats),
        "theta": session.theta
    }
    response = generate_plan(state)

    return {
        "insight": response
    }