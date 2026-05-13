"""
modules/llm.py  —  Phase 1 upgrade
LLM backend: Groq  |  Model: llama-3.3-70b-versatile
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment.")

client = Groq(api_key=GROQ_API_KEY)
# llama3-70b-8192 was decommissioned Aug 2025 — replaced with successor
MODEL  = 'llama-3.3-70b-versatile'

# ── Fixed bookend questions ───────────────────────────────────────────────────
FIRST_QUESTION = "Tell me about yourself and walk me through your background."
LAST_QUESTION  = "Where do you see yourself in 5 years, and how does this role fit into that vision?"

# ── Role-specific system prompts ──────────────────────────────────────────────
ROLE_SYSTEM_PROMPTS = {
    "sde": (
        "You are a senior Software Development Engineer conducting a technical interview. "
        "Focus on data structures, algorithms, system design, coding practices, and past engineering projects."
    ),
    "data scientist": (
        "You are a senior Data Scientist conducting a technical interview. "
        "Focus on ML/DL concepts, model evaluation, feature engineering, statistics, and past ML projects."
    ),
    "ml engineer": (
        "You are a senior ML Engineer conducting a technical interview. "
        "Focus on model deployment, MLOps, pipelines, scalability, and production ML systems."
    ),
    "pm": (
        "You are a senior Product Manager conducting a behavioral and strategy interview. "
        "Focus on product thinking, prioritization, stakeholder management, and past product decisions."
    ),
    "default": (
        "You are a professional interviewer conducting a structured interview. "
        "Ask thoughtful questions relevant to the candidate's background and target role."
    ),
}

def _get_system_prompt(job_role: str) -> str:
    key = job_role.lower().strip()
    for role_key in ROLE_SYSTEM_PROMPTS:
        if role_key in key:
            return ROLE_SYSTEM_PROMPTS[role_key]
    return ROLE_SYSTEM_PROMPTS["default"]


# ── Question Generator ────────────────────────────────────────────────────────
def generate_questions(name: str, job_role: str, experience: str,
                       skills: str, resume_text: str = '', jd_text: str = '',
                       difficulty: str = 'Medium',
                       num_questions: int = 2) -> list[str]:
    """
    Generate role-specific interview questions personalised to the candidate.
    Returns a list: [FIRST_QUESTION, ...generated..., LAST_QUESTION]
    """
    system_prompt = _get_system_prompt(job_role)

    resume_section = ""
    if resume_text:
        resume_section = f"\nResume Highlights:\n{resume_text[:2000]}\n"

    jd_section = ""
    if jd_text:
        jd_section = f"\nJob Description Context:\n{jd_text[:1000]}\n"

    diff_prompts = {
        "Easy": "Keep questions foundational, focusing on core concepts and straightforward past experiences.",
        "Medium": "Include a balance of practical implementation challenges and conceptual understanding.",
        "Advance": "Focus heavily on complex edge cases, system bottlenecks, in-depth architectural decisions, and deep domain expertise."
    }
    diff_instruction = diff_prompts.get(difficulty, diff_prompts["Medium"])

    user_prompt = f"""Generate exactly {num_questions} interview questions for the following candidate.
Return ONLY a numbered list. No explanation, no preamble, no trailing text.

Candidate Profile:
- Name: {name}
- Target Role: {job_role}
- Experience: {experience}
- Key Skills: {skills}
- Difficulty: {difficulty}
{resume_section}
{jd_section}
Rules:
- Difficulty Tuning: {diff_instruction}
- Mix technical and behavioral questions in equal proportion.
- Make each question specific to their background and the provided Job Description context — avoid generic questions.
- Each question should be a single clear sentence."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
        )
        raw   = (response.choices[0].message.content or "").strip()
        lines = [l.strip() for l in raw.split('\n') if l.strip()]

        questions = []
        for line in lines:
            if line and line[0].isdigit():
                q = line.split('.', 1)[-1].strip()
                q = q.split(')', 1)[-1].strip()
                if q:
                    questions.append(q)

        questions = questions[:num_questions]

        # Pad with fallback if parsing returned fewer than expected
        fallbacks = [
            f"Describe a challenging {job_role} project you've worked on.",
            f"How do you stay updated with the latest trends in {job_role}?",
        ]
        while len(questions) < num_questions:
            questions.append(fallbacks[len(questions) % len(fallbacks)])

        return [FIRST_QUESTION] + questions + [LAST_QUESTION]

    except Exception as e:
        print(f"[generate_questions] Error: {e}")
        return [
            FIRST_QUESTION,
            f"Tell me about a challenging project you've worked on as a {job_role}.",
            LAST_QUESTION,
        ]


# ── Follow-up Question Generator ─────────────────────────────────────────────
def generate_followup(question: str, answer: str, job_role: str) -> str | None:
    """
    Given the candidate's answer, decide if a follow-up is warranted.
    Returns a follow-up question string, or None if not needed.
    """
    system_prompt = _get_system_prompt(job_role)

    user_prompt = f"""You are interviewing a candidate for: {job_role}

Original Question: {question}
Candidate Answer: {answer}

Decide: does this answer warrant a follow-up probe?
- If the answer is vague, incomplete, or raises an interesting point worth exploring → return ONE concise follow-up question.
- If the answer is complete and sufficient → return exactly: NO_FOLLOWUP

Return ONLY the follow-up question or NO_FOLLOWUP. Nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.5,
        )
        result = (response.choices[0].message.content or "").strip()
        if result == "NO_FOLLOWUP" or not result:
            return None
        return result
    except Exception as e:
        print(f"[generate_followup] Error: {e}")
        return None


# ── Answer Evaluator ──────────────────────────────────────────────────────────
def evaluate_answer(question: str, answer: str, job_role: str,
                    followup: str = '', followup_answer: str = '') -> dict:
    """
    Evaluate candidate's answer. Returns a structured dict with score and feedback.
    Optionally includes follow-up Q&A in evaluation context.
    """
    if not answer or len(answer.strip()) < 5:
        return {
            "score":       0,
            "score_str":   "0/10",
            "strength":    "No answer detected.",
            "improvement": "Please speak clearly into the mic.",
            "raw":         "Score: 0/10\nStrength: No answer detected.\nImprovement: Please speak clearly into the mic.",
        }

    followup_section = ""
    if followup and followup_answer:
        followup_section = f"\nFollow-up Question: {followup}\nFollow-up Answer: {followup_answer}\n"

    system_prompt = _get_system_prompt(job_role)

    user_prompt = f"""You are evaluating a candidate for the role of: {job_role}

Question: {question}
Candidate Answer: {answer}{followup_section}

Evaluate and respond ONLY in this exact format — no extra text:
Score: X/10
Strength: <one sentence about what they did well>
Improvement: <one specific, actionable suggestion>
Detail: <two sentences of deeper feedback>"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_evaluation(raw)

    except Exception as e:
        print(f"[evaluate_answer] Error: {e}")
        fallback = "Score: 5/10\nStrength: Answer provided.\nImprovement: Technical issues prevented detailed evaluation.\nDetail: Please retry."
        return _parse_evaluation(fallback)


def _parse_evaluation(raw: str) -> dict:
    """Parse structured evaluation text into a dict."""
    result = {"raw": raw, "score": 5, "score_str": "5/10",
              "strength": "", "improvement": "", "detail": ""}

    for line in raw.split('\n'):
        if ':' not in line:
            continue
        key, _, value = line.partition(':')
        key   = key.strip().lower()
        value = value.strip()

        if key == "score":
            result["score_str"] = value
            try:
                result["score"] = int(value.split('/')[0])
            except ValueError:
                pass
        elif key == "strength":
            result["strength"] = value
        elif key == "improvement":
            result["improvement"] = value
        elif key == "detail":
            result["detail"] = value

    return result


# ── Final Summary Generator ───────────────────────────────────────────────────
def generate_final_summary(results: list[dict], job_role: str) -> dict:
    """
    Generate overall interview summary from all Q&A results.
    Each result dict should have: question, answer, feedback (dict or str).
    Returns a structured summary dict.
    """
    def feedback_str(r):
        fb = r.get('feedback', '')
        if isinstance(fb, dict):
            return fb.get('raw', '')
        return str(fb)

    all_feedback = '\n\n'.join([
        f"Q: {r['question']}\nA: {r['answer']}\nFeedback: {feedback_str(r)}"
        for r in results
    ])

    # Compute average score if evaluations are structured
    scores = []
    for r in results:
        fb = r.get('feedback', {})
        if isinstance(fb, dict) and 'score' in fb:
            scores.append(fb['score'])
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    user_prompt = f"""You evaluated a complete mock interview for the role of: {job_role}

Interview Transcript:
{all_feedback}

{'Computed average score across all questions: ' + str(avg_score) + '/10' if avg_score else ''}

Provide a final summary in this EXACT format — no extra text:
Overall Score: X/10
Top Strength: <one sentence>
Top Area to Improve: <one specific, actionable sentence>
Weak Topics: <comma-separated list of 2-3 topic areas to work on>
Final Tip: <one motivating, specific sentence>"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert interview coach giving final feedback."},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.4,
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_summary(raw, avg_score)

    except Exception as e:
        print(f"[generate_final_summary] Error: {e}")
        fallback = "Overall Score: 5/10\nTop Strength: Completed the interview.\nTop Area to Improve: Practice clearer answers.\nWeak Topics: Communication, Technical Depth\nFinal Tip: Keep practicing — consistency builds confidence!"
        return _parse_summary(fallback, avg_score)


def _parse_summary(raw: str, avg_score=None) -> dict:
    """Parse summary text into a structured dict."""
    result = {
        "raw": raw,
        "overall_score": avg_score or 5,
        "overall_score_str": f"{avg_score}/10" if avg_score else "5/10",
        "top_strength": "",
        "top_area_to_improve": "",
        "weak_topics": [],
        "final_tip": "",
    }
    for line in raw.split('\n'):
        if ':' not in line:
            continue
        key, _, value = line.partition(':')
        key   = key.strip().lower().replace(' ', '_')
        value = value.strip()

        if key == "overall_score":
            result["overall_score_str"] = value
            try:
                result["overall_score"] = float(value.split('/')[0])
            except ValueError:
                pass
        elif key == "top_strength":
            result["top_strength"] = value
        elif key == "top_area_to_improve":
            result["top_area_to_improve"] = value
        elif key == "weak_topics":
            result["weak_topics"] = [t.strip() for t in value.split(',')]
        elif key == "final_tip":
            result["final_tip"] = value

    return result