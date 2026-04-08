from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid, copy

from sqlalchemy import create_engine


from src.llm_questions import build_graph
from src.llm_insight import generate_plan
from src.irt_lite import IRTEngine
from src.db_models import get_db, SessionModel, QuestionModel, AttemptModel, ModelInsight, Session
from src.schemas import SessionCreate, GenerateInsightRequest, GenerateQuestionRequest, QuestionResponse, SubmitAnswerRequest


# Router
router = APIRouter()
graph = build_graph()

# -------------------------
# Endpoints
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
        "Q": None,
        "History": session.topic_distribution,
        "Evaluator_Feedback": ""
    }

    try:
        q_state = graph.invoke(state)['Q']
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