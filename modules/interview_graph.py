"""
modules/interview_graph.py — Phase 2: LangGraph Multi-Agent Interview Flow

StateGraph with three nodes:
  1. InterviewerAgent  — asks questions via TTS, captures answers via STT,
                         optionally asks one follow-up
  2. EvaluatorAgent    — scores the answer via LLM and stores feedback
  3. SummaryNode       — generates final summary and saves session JSON

Entry point:
    run_interview(profile: dict) -> dict   (returns final_summary)
"""

import os
import json
from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, END

# ── Sibling module imports ────────────────────────────────────────────────────
from modules.llm import (
    generate_questions,
    generate_followup,
    evaluate_answer,
    generate_final_summary,
)
from modules.stt import transcribe
from modules.tts import speak


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class InterviewState(TypedDict):
    profile: dict               # from resume_to_profile()
    questions: list[str]        # generated question bank
    current_index: int          # which question we're on
    results: list[dict]         # per-question results
    final_summary: dict         # filled by SummaryNode
    status: str                 # "interviewing" | "evaluating" | "done"


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 1 — INTERVIEWER AGENT
# ══════════════════════════════════════════════════════════════════════════════

def interviewer_agent(state: InterviewState) -> dict:
    """
    1. Pick the current question from the bank.
    2. Speak it via TTS.
    3. Capture the candidate's answer via STT.
    4. Optionally ask ONE follow-up and capture its answer.
    5. Store raw Q&A data in the results list for the EvaluatorAgent.
    """
    idx       = state["current_index"]
    questions = state["questions"]
    profile   = state["profile"]
    question  = questions[idx]

    q_num = idx + 1
    total = len(questions)

    # ── Speak the question ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  QUESTION {q_num}/{total}")
    print(f"{'='*60}")
    print(f"  Q: {question}\n")

    speak(f"Question {q_num}. {question}")

    # ── Capture answer via STT ────────────────────────────────────────────────
    print("[INTERVIEWER] Listening for your answer...")
    try:
        answer = transcribe()
    except Exception as e:
        print(f"[INTERVIEWER] STT error: {e}")
        answer = ""

    if not answer.strip():
        answer = "[No answer detected]"
    print(f"[INTERVIEWER] Heard: {answer[:150]}{'...' if len(answer) > 150 else ''}")

    # ── Follow-up logic (max 1 per question) ──────────────────────────────────
    followup_q      = None
    followup_answer = ""

    if answer and not answer.startswith("["):
        job_role = profile.get("job_role", "Software Engineer")
        try:
            followup_q = generate_followup(question, answer, job_role)
        except Exception as e:
            print(f"[INTERVIEWER] Follow-up generation error: {e}")

    if followup_q:
        print(f"\n[INTERVIEWER] Follow-up: {followup_q}")
        speak(f"Follow-up question: {followup_q}")

        print("[INTERVIEWER] Listening for follow-up answer...")
        try:
            followup_answer = transcribe()
        except Exception as e:
            print(f"[INTERVIEWER] Follow-up STT error: {e}")
            followup_answer = ""

        if not followup_answer.strip():
            followup_answer = "[No follow-up answer detected]"
        print(f"[INTERVIEWER] Follow-up answer: {followup_answer[:150]}{'...' if len(followup_answer) > 150 else ''}")

    # ── Build result entry (feedback will be filled by EvaluatorAgent) ────────
    result_entry = {
        "question":        question,
        "answer":          answer,
        "followup":        followup_q,
        "followup_answer": followup_answer,
        "feedback":        {},          # placeholder — filled by evaluator
    }

    # Append to results list
    updated_results = list(state["results"]) + [result_entry]

    return {
        "results": updated_results,
        "status":  "evaluating",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 2 — EVALUATOR AGENT
# ══════════════════════════════════════════════════════════════════════════════

def evaluator_agent(state: InterviewState) -> dict:
    """
    1. Evaluate the latest answer using the LLM.
    2. Store structured feedback in the result entry.
    3. Advance current_index.
    4. Set status back to 'interviewing' (routing logic decides next step).
    """
    profile = state["profile"]
    results = list(state["results"])         # shallow copy for mutation
    idx     = state["current_index"]

    # The latest result is the one we just captured
    latest = results[-1]

    job_role = profile.get("job_role", "Software Engineer")

    print(f"\n[EVALUATOR] Scoring answer for Q{idx + 1}...")

    try:
        feedback = evaluate_answer(
            question=latest["question"],
            answer=latest["answer"],
            job_role=job_role,
            followup=latest.get("followup") or "",
            followup_answer=latest.get("followup_answer") or "",
        )
    except Exception as e:
        print(f"[EVALUATOR] Evaluation error: {e}")
        feedback = {
            "score":       5,
            "score_str":   "5/10",
            "strength":    "Answer recorded.",
            "improvement": "Evaluation unavailable due to error.",
            "detail":      "",
            "raw":         "",
        }

    # Store feedback in the result
    latest["feedback"] = feedback
    results[-1] = latest

    print(f"[EVALUATOR] Score: {feedback.get('score_str', '?')}  |  "
          f"Strength: {feedback.get('strength', '')}")
    print(f"[EVALUATOR] Improve: {feedback.get('improvement', '')}")

    # Advance to next question
    new_index = idx + 1

    return {
        "results":       results,
        "current_index": new_index,
        "status":        "interviewing",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 3 — SUMMARY NODE
# ══════════════════════════════════════════════════════════════════════════════

def summary_node(state: InterviewState) -> dict:
    """
    1. Generate final interview summary via LLM.
    2. Save the complete session to a JSON file.
    3. Return final_summary dict.
    """
    profile  = state["profile"]
    results  = state["results"]
    job_role = profile.get("job_role", "Software Engineer")

    print(f"\n{'='*60}")
    print("  GENERATING FINAL SUMMARY")
    print(f"{'='*60}")

    try:
        final_summary = generate_final_summary(results, job_role)
    except Exception as e:
        print(f"[SUMMARY] Error generating summary: {e}")
        final_summary = {
            "overall_score":          5,
            "overall_score_str":      "5/10",
            "top_strength":           "Completed the interview.",
            "top_area_to_improve":    "Practice more.",
            "weak_topics":            [],
            "final_tip":              "Keep practicing!",
            "raw":                    "",
        }

    # ── Print summary to console ─────────────────────────────────────────────
    print(f"\n  Overall Score   : {final_summary.get('overall_score_str', '?')}")
    print(f"  Top Strength    : {final_summary.get('top_strength', '')}")
    print(f"  Top Improvement : {final_summary.get('top_area_to_improve', '')}")
    print(f"  Weak Topics     : {', '.join(final_summary.get('weak_topics', []))}")
    print(f"  Final Tip       : {final_summary.get('final_tip', '')}")

    # ── Speak a brief closing ─────────────────────────────────────────────────
    closing_text = (
        f"Interview complete. Your overall score is "
        f"{final_summary.get('overall_score_str', 'not available')}. "
        f"{final_summary.get('final_tip', 'Great job!')}"
    )
    try:
        speak(closing_text)
    except Exception:
        pass

    # ── Save session to JSON ──────────────────────────────────────────────────
    session_data = {
        "timestamp":     datetime.now().isoformat(),
        "profile":       _make_serialisable(profile),
        "questions":     state["questions"],
        "results":       _make_serialisable(results),
        "final_summary": _make_serialisable(final_summary),
    }

    sessions_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sessions",
    )
    os.makedirs(sessions_dir, exist_ok=True)

    filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(sessions_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        print(f"\n[SUMMARY] Session saved → {filepath}")
    except Exception as e:
        print(f"[SUMMARY] Failed to save session JSON: {e}")

    return {
        "final_summary": final_summary,
        "status":        "done",
    }


# ── Utility: make dicts JSON-serialisable ─────────────────────────────────────

def _make_serialisable(obj):
    """Recursively convert non-serialisable types to strings."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def _after_evaluator(state: InterviewState) -> str:
    """
    Conditional edge after EvaluatorAgent:
      - If more questions remain → loop back to InterviewerAgent
      - Otherwise               → proceed to SummaryNode
    """
    if state["current_index"] < len(state["questions"]):
        return "interviewer"
    return "summary"


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Flow:
        START → interviewer → evaluator ──┬──→ interviewer  (loop)
                                          └──→ summary → END
    """
    graph = StateGraph(InterviewState)

    # Add nodes
    graph.add_node("interviewer", interviewer_agent)
    graph.add_node("evaluator",   evaluator_agent)
    graph.add_node("summary",     summary_node)

    # Set entry point
    graph.set_entry_point("interviewer")

    # Edges
    graph.add_edge("interviewer", "evaluator")           # always goes to evaluator

    graph.add_conditional_edges(                          # evaluator → loop or finish
        "evaluator",
        _after_evaluator,
        {
            "interviewer": "interviewer",
            "summary":     "summary",
        },
    )

    graph.add_edge("summary", END)                       # summary → done

    return graph.compile()


# Compile once at module level
_compiled_graph = _build_graph()


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_interview(profile: dict, num_questions: int = 2) -> dict:
    """
    Run a full multi-agent mock interview.

    Args:
        profile:       dict from resume_to_profile() containing name, job_role,
                       experience, skills, resume_text, etc.
        num_questions: number of middle questions to generate (bookend questions
                       are added automatically).

    Returns:
        final_summary dict with overall_score, top_strength,
        top_area_to_improve, weak_topics, final_tip, etc.
    """
    name       = profile.get("name",       "Candidate")
    job_role   = profile.get("job_role",    "Software Engineer")
    experience = profile.get("experience",  "Fresher")
    skills     = profile.get("skills",      "")
    resume_text = profile.get("resume_text", "")

    print(f"\n{'='*60}")
    print(f"  AI MOCK INTERVIEW — LangGraph Multi-Agent Flow")
    print(f"{'='*60}")
    print(f"  Candidate : {name}")
    print(f"  Role      : {job_role}")
    print(f"  Experience: {experience}")
    print(f"{'='*60}\n")

    # ── Generate question bank ────────────────────────────────────────────────
    print("[SETUP] Generating interview questions...")
    questions = generate_questions(
        name=name,
        job_role=job_role,
        experience=experience,
        skills=skills,
        resume_text=resume_text,
        num_questions=num_questions,
    )
    print(f"[SETUP] {len(questions)} questions ready.\n")

    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    print()

    # ── Greet candidate ───────────────────────────────────────────────────────
    greeting = (
        f"Welcome {name}. This is your mock interview for the {job_role} role. "
        f"I will ask you {len(questions)} questions. Let's begin."
    )
    speak(greeting)

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state: InterviewState = {
        "profile":       profile,
        "questions":     questions,
        "current_index": 0,
        "results":       [],
        "final_summary": {},
        "status":        "interviewing",
    }

    # ── Run the graph ─────────────────────────────────────────────────────────
    final_state = _compiled_graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print("  INTERVIEW SESSION COMPLETE")
    print(f"{'='*60}\n")

    return final_state["final_summary"]


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — quick test entry
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    from modules.resume_parser import resume_to_profile

    print("\n" + "=" * 60)
    print("  LANGGRAPH INTERVIEW — QUICK START")
    print("=" * 60)
    print("  [1] Upload resume (PDF/DOCX/TXT/MD)")
    print("  [2] Manual profile")
    print("=" * 60)
    choice = input("  Choice: ").strip()

    if choice == "1":
        path = input("  Resume path: ").strip().strip('"')
        print("  Parsing resume...")
        profile = resume_to_profile(path)
        print(f"  ✓ Parsed: {profile['name']} | {profile['job_role']}")
    else:
        profile = {
            "name":        input("  Name     : ").strip(),
            "job_role":    input("  Job Role : ").strip(),
            "experience":  input("  Experience: ").strip(),
            "skills":      input("  Skills   : ").strip(),
            "resume_text": "",
        }

    num_q = input("  Number of questions (default 2): ").strip()
    num_q = int(num_q) if num_q.isdigit() else 2

    summary = run_interview(profile, num_questions=num_q)
    print("\n[DONE] Final summary returned:")
    print(json.dumps(summary, indent=2))
