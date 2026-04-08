import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from src.schemas import UserSession
from src.generator_llm import generate_question_node
from src.evaluator_llm import evaluate_question_node


load_dotenv()
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))

#  ────────────────────────────────────────────────────────────────────────────────────
#  LangGraph conditional edge
def route_after_eval(state: dict) -> str:
    """
    Returns the name of the next node:
        "accept"       → question passed, add to bank
        "regenerate"   → retry with evaluator feedback
        "discard"      → max retries hit, send to human review
    """
    eval_result = state.get("EvalResult")
    retries     = state.get("RetryCount", 0)

    if eval_result and eval_result.passed:
        return "accept"

    if retries >= MAX_RETRIES:
        return "discard"

    # Increment retry counter before looping back
    state["RetryCount"] = retries + 1
    return "regenerate"



def build_graph():
    # ── Nodes ────────────────────────────────────────────────────────────────────
    nodes = [
        ("gen_q",      generate_question_node),
        ("evaluate_q", evaluate_question_node),
    ]

    # ── Linear edges ─────────────────────────────────────────────────────────────
    edges = [
        (START,    "gen_q"),
        ("gen_q",  "evaluate_q"),
    ]

    # ── Conditional edges ────────────────────────────────────────────────────────
    conditional_edges = [
        ("evaluate_q", route_after_eval, {
            "accept":     END,       
            "discard":    END,       
            "regenerate": "gen_q",  
        })
    ]

    # ── Build ─────────────────────────────────────────────────────────────────────
    builder = StateGraph(UserSession)

    for name, fn in nodes:
        builder.add_node(name, fn)

    for src, dst in edges:
        builder.add_edge(src, dst)

    for src, fn, mapping in conditional_edges:
        builder.add_conditional_edges(src, fn, mapping)

    graph = builder.compile()
    return graph

if __name__ == '__main__':pass