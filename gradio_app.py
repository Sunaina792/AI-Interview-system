"""
gradio_app.py — AI Smart Mock Interview System
Premium Wizard-style UI
"""
import gradio as gr
import cv2
import numpy as np
import os
import sys
import json
import time
import threading
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.face_landmarks import FaceLandmarkExtractor
from modules.expression_detection import ExpressionDetector
from modules.eye_contact import EyeContactDetector
from modules.head_pose import HeadPoseEstimator
from modules.stt import transcribe as stt_transcribe
from modules.tts import speak, speak_async
from modules.llm import generate_questions, generate_followup, evaluate_answer, generate_final_summary
from modules.resume_parser import resume_to_profile, get_resume_context_for_llm
from modules.face_verify import extract_face_embedding, verify_face

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_AVATAR_CANDIDATES = [
    os.path.join(_BASE_DIR, "interview.jpg"),
    os.path.join(_BASE_DIR, "assets", "avatar.png"),
]
AI_AVATAR_PATH = next((p for p in _AVATAR_CANDIDATES if os.path.exists(p)), None)
if AI_AVATAR_PATH is None:
    # fallback: generate a blank placeholder saved alongside the script
    _blank = np.zeros((400, 400, 3), dtype=np.uint8)
    AI_AVATAR_PATH = os.path.join(_BASE_DIR, "dummy_avatar.png")
    cv2.imwrite(AI_AVATAR_PATH, _blank)

# ── Globals for webcam analysis ───────────────────────────────────────────────
landmark_extractor = None
expr_detector = None
eye_detector = None
head_estimator = None

score_history = deque(maxlen=200)
expr_counts = {}

WEIGHTS = {
    "eye_contact": 30,
    "expression": 25,
    "head_stability": 25,
    "nervousness": 20,
}

# ── Interview Session State ───────────────────────────────────────────────────
interview_state = {
    "profile": None,
    "jd_text": "",
    "questions": [],
    "current_index": 0,
    "results": [],
    "status": "idle",
    "final_summary": None,
    "current_question": "",
    "reference_embedding": None,
    "start_time": 0,
    "time_limit": 6 * 60,  # 6 minutes default
    "report_generated": False,  # guard: prevent repeated LLM calls on timer ticks
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _init_detectors():
    global landmark_extractor, expr_detector, eye_detector, head_estimator
    if landmark_extractor is None:
        landmark_extractor = FaceLandmarkExtractor()
        expr_detector = ExpressionDetector(fps=20)
        eye_detector = EyeContactDetector()
        head_estimator = HeadPoseEstimator()

def compute_confidence(expr_r, eye_r, head_r):
    eye_s = eye_r.get("score", 0)
    expr_s = expr_r.get("expression_score", 0)
    head_s = head_r.get("stability_score", 0)
    nerv_s = expr_r.get("nervousness_score", 0)
    nerv_c = max(0, 100 - nerv_s)
    return min(100, max(0, int(
        eye_s  * WEIGHTS["eye_contact"]    / 100 +
        expr_s * WEIGHTS["expression"]     / 100 +
        head_s * WEIGHTS["head_stability"] / 100 +
        nerv_c * WEIGHTS["nervousness"]    / 100
    )))

def confidence_label(score):
    if score >= 80: return "High"
    if score >= 60: return "Moderate"
    if score >= 40: return "Low"
    return "Very Low"

# ── Phase 1: Permissions & Selfie ──────────────────────────────────────────────
def capture_identity(frame):
    if frame is None:
        return gr.update(value="⚠️ No face captured. Please allow camera and snap.", visible=True), gr.update()
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    try:
        emb = extract_face_embedding(frame_bgr)
        if emb is None:
            return gr.update(value="⚠️ No face detected in image. Please center your face.", visible=True), gr.update()
        
        interview_state["reference_embedding"] = emb
        return gr.update(value="✅ Identity verified! Moving to Phase 2...", visible=True), gr.update(visible=False, value=True)
    except Exception as e:
        return gr.update(value=f"⚠️ Error: {e}", visible=True), gr.update()

def proceed_to_phase2(dummy):
    if dummy:
        # returns updates to make Screen 2 visible and Screen 1 hidden
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(), gr.update()

# ── Phase 2: Setup (Resume + JD) ──────────────────────────────────────────────
def parse_resume_file(file):
    if file is None:
        return "No file uploaded.", "", "", "", ""
    try:
        profile = resume_to_profile(file.name)
        interview_state["profile"] = profile
        return profile.get("name",""), profile.get("job_role",""), profile.get("experience",""), profile.get("skills",""), "✅ Resume Parsed!"
    except Exception as e:
        return "", "", "", "", f"Error parsing: {e}"

def start_interview_flow(jd_text, difficulty, num_q):
    profile = interview_state["profile"]
    if not profile:
        profile = {"name": "Candidate", "job_role": "Software Engineer", "experience": "Fresher", "skills": "Python"}
        interview_state["profile"] = profile
    
    interview_state["jd_text"] = jd_text
    num_questions = int(num_q)
    
    name = profile.get("name", "Candidate")
    role = profile.get("job_role", "Software Engineer")
    exp = profile.get("experience", "Fresher")
    skills = profile.get("skills", "")
    resume_text = get_resume_context_for_llm(profile)
    
    # Generate Questions
    questions = generate_questions(
        name=name, job_role=role, experience=exp, skills=skills,
        resume_text=resume_text, jd_text=jd_text, difficulty=difficulty, num_questions=num_questions
    )
    
    interview_state["questions"] = questions
    interview_state["current_index"] = 0
    interview_state["results"] = []
    interview_state["status"] = "interviewing"
    
    curr_q = questions[0]
    interview_state["current_question"] = curr_q
    
    # AI starts talking — speak_async fires immediately in background
    speak_async(f"Hello {name}. Let's begin the mock interview. First question: {curr_q}")
    
    interview_state["start_time"] = time.time()
    
    status_text = f"## Question 1 of {len(questions)}\n\n**{curr_q}**"
    
    return gr.update(visible=False), gr.update(visible=True), status_text

# ── Phase 3: Interview Room ───────────────────────────────────────────────────
def analyze_frame(frame):
    """Process a single webcam frame and return annotated frame + metrics."""
    if frame is None:
        return None
    _init_detectors()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    lm_result = landmark_extractor.extract(frame_bgr)
    out = lm_result["annotated_frame"].copy()

    # Proctoring Check
    ref_emb = interview_state.get("reference_embedding")
    if ref_emb is not None:
        ver = verify_face(ref_emb, frame_bgr)
        if ver["alert"]:
            cv2.putText(out, ver["alert"], (10, min(160, out.shape[0]-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if lm_result["face_detected"]:
        kp = lm_result["key_points"]
        landmarks = lm_result["landmarks"]
        expr_result = expr_detector.detect(kp, frame_bgr.shape)
        eye_result = eye_detector.detect(kp, frame_bgr.shape)
        head_result = head_estimator.detect(landmarks, frame_bgr.shape)
        confidence = compute_confidence(expr_result, eye_result, head_result)

        label = confidence_label(confidence)
        color = (0, 220, 0) if confidence >= 70 else (0, 200, 255) if confidence >= 40 else (0, 80, 255)

        cv2.putText(out, f"Score: {confidence}/100 [{label}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(out, f"Expr: {expr_result.get('expression', 'N/A')}  Nerv: {expr_result.get('nervousness_score', 0)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(out, f"Gaze: {eye_result.get('gaze_direction', 'N/A')}  EyeC: {eye_result.get('eye_contact_pct', 0)}%",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(out, f"Head: {head_result.get('direction', 'N/A')}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out_rgb

def get_timer():
    if interview_state["start_time"] == 0:
        return "Timer: 00:00"
    elapsed = int(time.time() - interview_state["start_time"])
    m, s = divmod(elapsed, 60)
    return f"⏱️ Elapsed Time: {m:02d}:{s:02d}"

def submit_audio_answer(audio_path):
    if not audio_path:
        return "Please record an answer.", gr.update()
    
    # Transcribe
    ans_text = stt_transcribe(audio_path)
    if not ans_text:
        ans_text = "(Audio unclear or empty)"
    
    # Feedback
    idx = interview_state["current_index"]
    questions = interview_state["questions"]
    question = questions[idx]
    role = interview_state["profile"].get("job_role", "Role")
    
    # follow up (skipping to keep it 5 mins unless asked)
    feedback = evaluate_answer(question=question, answer=ans_text, job_role=role)
    
    interview_state["results"].append({
        "question": question, "answer": ans_text, "feedback": feedback
    })
    
    # Next question
    interview_state["current_index"] += 1
    if interview_state["current_index"] < len(questions):
        next_q = questions[interview_state["current_index"]]
        interview_state["current_question"] = next_q
        speak_async(f"Question {interview_state['current_index']+1}. {next_q}")
        
        status_md = f"## Question {interview_state['current_index']+1} of {len(questions)}\n\n**{next_q}**"
        # Reset audio input
        return status_md, None
    else:
        # Done
        interview_state["status"] = "done"
        return "## 🎉 Interview Complete! Generating exact report...", None

def check_done():
    if interview_state["status"] == "done":
        # Guard: only generate the report once, not on every timer tick
        if interview_state["report_generated"]:
            return gr.update(), gr.update(), gr.update()

        interview_state["report_generated"] = True

        role = interview_state["profile"].get("job_role", "Role")
        summary = generate_final_summary(interview_state["results"], role)
        interview_state["final_summary"] = summary

        # Format Report
        rep = f"# 📊 Final Report\n\nOverall Score: {summary.get('overall_score_str')}\n\n"
        rep += f"**Top Strength:** {summary.get('top_strength')}\n\n"
        rep += f"**To Improve:** {summary.get('top_area_to_improve')}\n\n"
        rep += "### Per Question Responses\n"
        for i, res in enumerate(interview_state["results"]):
            fb = res.get('feedback', {})
            score_str = fb.get('score_str', 'N/A') if isinstance(fb, dict) else 'N/A'
            tip = fb.get('improvement', 'N/A') if isinstance(fb, dict) else str(fb)
            rep += f"**Q{i+1}: {res['question']}**\n- Score: {score_str}\n- Answer: {res['answer']}\n- Tip: {tip}\n\n"

        return gr.update(visible=False), gr.update(visible=True), rep
    return gr.update(), gr.update(), gr.update()


# ── Full Gradio Layout ────────────────────────────────────────────────────────
CUSTOM_CSS = """
body { font-family: 'Inter', sans-serif !important; background: #0f172a; color: #f8fafc; }
.card-panel { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
.primary-btn { background: linear-gradient(135deg, #10b981, #059669) !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 8px !important; }
.text-box input, .text-box textarea { background: #0f172a !important; color: white !important; border-color: #334155 !important; }
h1, h2, h3 { color: #f8fafc !important; }
"""

with gr.Blocks() as app:
    
    # ━━ Screen 1: Permissions ━━
    with gr.Group(visible=True) as screen_permissions:
        gr.Markdown("# 🛡️ Interview Setup: Identity Verification\nPlease allow your camera to take a brief reference photo. This ensures identity proctoring during the interview.")
        with gr.Row():
            auth_cam = gr.Image(sources=["webcam"], label="Take an ID Selfie", elem_classes=["card-panel"])
        
        auth_btn = gr.Button("Capture Identity & Continue", elem_classes=["primary-btn"], size="lg")
        auth_status = gr.Markdown("")
        hidden_proceed_flag = gr.Checkbox(visible=False)
        
    # ━━ Screen 2: Setup (Resume + JD) ━━
    with gr.Group(visible=False) as screen_setup:
        gr.Markdown("# 📄 Interview Constraints\nUpload your resume and paste the Job Description for a highly tailored mock interview.")
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["card-panel"]):
                gr.Markdown("### 1. Resume")
                res_file = gr.File(label="Upload Resume (PDF/DOCX/TXT)")
                parse_btn = gr.Button("Parse Resume", variant="secondary")
                parse_msg = gr.Markdown()
                
            with gr.Column(scale=2, elem_classes=["card-panel"]):
                gr.Markdown("### Profile Preview")
                r_name = gr.Textbox(label="Name", interactive=False)
                r_role = gr.Textbox(label="Role", interactive=False)
                r_exp = gr.Textbox(label="Experience", interactive=False)
                r_skills = gr.Textbox(label="Skills", interactive=False)
                
        with gr.Row():
            with gr.Column(elem_classes=["card-panel"]):
                gr.Markdown("### 2. Job Description")
                jd_input = gr.Textbox(lines=6, placeholder="Paste JD here... (If left blank, questions will be general to your resume role)", label="Target JD")
                
        with gr.Row():
            difficulty_radio = gr.Radio(["Easy", "Medium", "Advance"], value="Medium", label="Difficulty Level")
            num_q_slider = gr.Slider(minimum=2, maximum=10, step=1, value=5, label="Number of Questions (4-6 min length = 5)")
            
        with gr.Row():
            start_btn = gr.Button("🏁 Enter Interview Room", elem_classes=["primary-btn"], size="lg")

    # ━━ Screen 3: Interview Room ━━
    with gr.Group(visible=False) as screen_interview:
        with gr.Row():
            gr.Markdown("# 🎙️ Interview Room")
            timer_html = gr.Markdown("⏱️ 00:00", elem_classes=["timer-display"])
            
        with gr.Row():
            # AI Avatar Column
            with gr.Column(scale=1, elem_classes=["card-panel"]):
                gr.Image(value=AI_AVATAR_PATH, interactive=False, label="AI Interviewer", height=300)
                q_text = gr.Markdown("### Loading question...", elem_id="question_box")
                
            # Candidate Webcam Column
            with gr.Column(scale=1, elem_classes=["card-panel"]):
                cam_feed = gr.Image(sources=["webcam"], streaming=True, label="Live Proctoring", height=300)
                
        with gr.Row(elem_classes=["card-panel"]):
            ans_audio = gr.Audio(sources=["microphone"], type="filepath", label="Record your answer")
            submit_ans_btn = gr.Button("📤 Submit & Next Question", elem_classes=["primary-btn"])
            
        check_timer = gr.Timer(1)
        
    # ━━ Screen 4: Report ━━
    with gr.Group(visible=False) as screen_report:
        gr.Markdown("# 🏆 Final AI Assessor Report")
        final_report_md = gr.Markdown("Compiling report...")
        restart_btn = gr.Button("Restart Application", variant="secondary")

    # ── Wiring ───────────────────────────────────────────────────────────────
    
    # Phase 1 logic
    auth_btn.click(fn=capture_identity, inputs=[auth_cam], outputs=[auth_status, hidden_proceed_flag])
    hidden_proceed_flag.change(fn=proceed_to_phase2, inputs=[hidden_proceed_flag], outputs=[screen_permissions, screen_setup])
    
    # Phase 2 logic
    parse_btn.click(fn=parse_resume_file, inputs=[res_file], outputs=[r_name, r_role, r_exp, r_skills, parse_msg])
    
    start_btn.click(
        fn=start_interview_flow, 
        inputs=[jd_input, difficulty_radio, num_q_slider], 
        outputs=[screen_setup, screen_interview, q_text]
    )
    
    # Phase 3 logic
    cam_feed.stream(fn=analyze_frame, inputs=[cam_feed], outputs=[cam_feed], stream_every=0.5)
    check_timer.tick(fn=get_timer, outputs=[timer_html]).then(
        fn=check_done, outputs=[screen_interview, screen_report, final_report_md]
    )
    
    submit_ans_btn.click(
        fn=submit_audio_answer, 
        inputs=[ans_audio], 
        outputs=[q_text, ans_audio]
    )
    
    restart_btn.click(fn=lambda: None, js="window.location.reload()")

if __name__ == "__main__":
    app.launch(server_port=7861, show_error=True, css=CUSTOM_CSS, theme=gr.themes.Monochrome())