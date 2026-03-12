# routes.py
from fastapi import APIRouter
from bson import ObjectId
from datetime import datetime, timezone

from app.models import AnswerSubmission
from app.database import sessions_collection, questions_collection
from app.adaptive import select_next_question, update_ability
from app.insight import analyze_session, generate_study_plan

router = APIRouter()


@router.post("/start-session")
def start_session():
    session = {
        "ability": 0.5,
        "answered_questions": [],
        "history": [],
        "completed": False,
        "started_at": datetime.now(timezone.utc),

        # ability trajectory over time
        "ability_trajectory": [0.5],

        # per-topic breakdown
        "topic_stats": {},

        # streaks
        "current_streak": 0,
        "max_streak": 0,

        # difficulty engagement
        "avg_difficulty_attempted": 0.0,
        "difficulty_trend": [],        # difficulty of each question in order

        # consistency score (low variance = consistent)
        "ability_variance": 0.0,

        "total_correct": 0,
        "total_attempted": 0,
    }

    result = sessions_collection.insert_one(session)
    return {"session_id": str(result.inserted_id)}


@router.get("/next-question/{session_id}")
def get_next_question(session_id: str):
    session = sessions_collection.find_one({"_id": ObjectId(session_id)})

    if not session:
        return {"error": "Session not found"}

    question = select_next_question(session["ability"], session["answered_questions"])

    if question is None:
        return {"message": "No more questions"}

    return {
        "question_id": str(question["_id"]),
        "question": question["question_text"],
        "options": question["options"],
        "difficulty": question["difficulty"]
    }


@router.post("/submit-answer")
def submit_answer(data: AnswerSubmission):
    session = sessions_collection.find_one({"_id": ObjectId(data.session_id)})
    if not session:
        return {"error": "Session not found"}

    question = questions_collection.find_one({"_id": ObjectId(data.question_id)})
    if not question:
        return {"error": "Question not found"}

    correct = data.answer == question["correct_answer"]
    difficulty = question["difficulty"]
    topic = question["topic"]

    new_ability = update_ability(session["ability"], correct, question)

    # --- update topic_stats ---
    topic_stats = session.get("topic_stats", {})
    t = topic_stats.get(topic, {"correct": 0, "attempted": 0, "avg_difficulty": 0.0})
    t["attempted"] += 1
    t["correct"] += 1 if correct else 0
    t["accuracy"] = round(t["correct"] / t["attempted"], 2)
    t["avg_difficulty"] = round(
        (t["avg_difficulty"] * (t["attempted"] - 1) + difficulty) / t["attempted"], 3
    )
    topic_stats[topic] = t

    # --- streaks ---
    current_streak = session.get("current_streak", 0)
    max_streak = session.get("max_streak", 0)
    current_streak = current_streak + 1 if correct else 0
    max_streak = max(max_streak, current_streak)

    # --- difficulty trend & avg ---
    difficulty_trend = session.get("difficulty_trend", [])
    difficulty_trend.append(difficulty)
    avg_difficulty = round(sum(difficulty_trend) / len(difficulty_trend), 3)

    # --- ability trajectory & variance ---
    trajectory = session.get("ability_trajectory", [0.5])
    trajectory.append(round(new_ability, 4))
    mean_ab = sum(trajectory) / len(trajectory)
    variance = round(sum((x - mean_ab) ** 2 for x in trajectory) / len(trajectory), 4)

    total_correct = session.get("total_correct", 0) + (1 if correct else 0)
    total_attempted = session.get("total_attempted", 0) + 1

    sessions_collection.update_one(
        {"_id": ObjectId(data.session_id)},
        {
            "$set": {
                "ability": new_ability,
                "topic_stats": topic_stats,
                "current_streak": current_streak,
                "max_streak": max_streak,
                "avg_difficulty_attempted": avg_difficulty,
                "difficulty_trend": difficulty_trend,
                "ability_trajectory": trajectory,
                "ability_variance": variance,
                "total_correct": total_correct,
                "total_attempted": total_attempted,
            },
            "$push": {
                "answered_questions": ObjectId(data.question_id),
                "history": {
                    "question_id": data.question_id,
                    "correct": correct,
                    "difficulty": difficulty,
                    "discrimination": question.get("discrimination", 1.0),
                    "topic": topic,
                    "ability_after": round(new_ability, 4),
                    "answered_at": datetime.now(timezone.utc)
                }
            }
        }
    )

    return {"correct": correct, "new_ability": new_ability}


@router.get("/finish-test/{session_id}")
def finish_test(session_id: str):
    session = sessions_collection.find_one({"_id": ObjectId(session_id)})
    if not session:
        return {"error": "Session not found"}

    topic_accuracy, weak_topics, strong_topics = analyze_session(session["history"])
    study_plan = generate_study_plan(session)   # pass full session now

    return {
        "final_ability": session["ability"],
        "ability_trajectory": session.get("ability_trajectory", []),
        "ability_variance": session.get("ability_variance", 0.0),
        "total_correct": session.get("total_correct", 0),
        "total_attempted": session.get("total_attempted", 0),
        "max_streak": session.get("max_streak", 0),
        "avg_difficulty_attempted": session.get("avg_difficulty_attempted", 0.0),
        "topic_stats": session.get("topic_stats", {}),
        "topic_accuracy": topic_accuracy,
        "weak_topics": weak_topics,
        "strong_topics": strong_topics,
        "study_plan": study_plan
    }