import os
from dotenv import load_dotenv
from datetime import datetime


from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import create_engine, Column, String, Float, TIMESTAMP, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

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


Base.metadata.create_all(bind=engine)