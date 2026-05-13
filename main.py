# main.py — AI Interview Confidence Analyzer
# Combines: Face Landmarks + Expression + Eye Contact + Head Pose + STT + LLM + TTS
# Phase 1: Full interview loop — STT + LLM eval + follow-up questions

import cv2
import sys
import os
import time
import threading
import numpy as np
from collections import deque

# Add modules folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from modules.face_landmarks       import FaceLandmarkExtractor
from modules.expression_detection import ExpressionDetector
from modules.eye_contact          import EyeContactDetector
from modules.head_pose            import HeadPoseEstimator
from modules.stt                  import transcribe
from modules.tts                  import speak
from modules.llm                  import (
    generate_questions,
    generate_followup,
    evaluate_answer,
    generate_final_summary,
    FIRST_QUESTION,
    LAST_QUESTION,
)
from modules.resume_parser import resume_to_profile, get_resume_context_for_llm


# CONFIDENCE SCORE WEIGHTS (must sum to 100)


WEIGHTS = {
    "eye_contact":    30,
    "expression":     25,
    "head_stability": 25,
    "nervousness":    20,
}

SCORE_HISTORY_LEN = 45   # ~1.5 seconds at 30fps


# CONFIDENCE CALCULATOR

def compute_confidence_score(expr_result, eye_result, head_result):
    eye_score  = eye_result.get("score",              0)
    expr_score = expr_result.get("expression_score",  0)
    head_score = head_result.get("stability_score",   0)
    nerv_score = expr_result.get("nervousness_score", 0)

    nerv_contribution = max(0, 100 - nerv_score)

    final = (
        eye_score         * WEIGHTS["eye_contact"]    / 100 +
        expr_score        * WEIGHTS["expression"]     / 100 +
        head_score        * WEIGHTS["head_stability"] / 100 +
        nerv_contribution * WEIGHTS["nervousness"]    / 100
    )
    return min(100, max(0, int(final)))


def confidence_label(score):
    if score >= 80: return "High",     (0, 220, 0)
    if score >= 60: return "Moderate", (0, 200, 150)
    if score >= 40: return "Low",      (0, 165, 255)
    return "Very Low", (0, 60, 255)


# DRAW DASHBOARD PANEL



def draw_dashboard(frame, expr_result, eye_result, head_result,
                   confidence, score_history, current_question="", status=""):
    h, w = frame.shape[:2]

    panel_x = w - 260
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    x  = panel_x
    y  = 30
    dy = 28

    def put(text, color=(220, 220, 220), scale=0.55, bold=1):
        nonlocal y
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold)
        y += dy

    put("CONFIDENCE ANALYZER", (255, 255, 255), 0.55, 2)
    put("-" * 28, (80, 80, 80))

    label, color = confidence_label(confidence)
    put(f"SCORE: {confidence}/100", color, 0.75, 2)
    put(f"Level: {label}", color)

    bar_w = int((confidence / 100) * 230)
    cv2.rectangle(frame, (x, y),       (x + 230, y + 12), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y),       (x + bar_w, y + 12), color,       -1)
    y += 22
    put("-" * 28, (80, 80, 80))

    expr     = expr_result.get("expression",        "N/A")
    nerv     = expr_result.get("nervousness_score",  0)
    blink_r  = expr_result.get("blink_rate",         0)
    expr_col = (0, 220, 0) if expr == "Happy" else (0, 165, 255) if nerv > 40 else (220, 220, 220)
    put("EXPRESSION", (180, 180, 255), 0.5, 1)
    put(f"  {expr}", expr_col)
    put(f"  Nerv: {nerv}/100  Blink:{blink_r}/m")
    put("-" * 28, (80, 80, 80))

    gaze    = eye_result.get("gaze_direction",  "N/A")
    eye_pct = eye_result.get("eye_contact_pct", 0)
    eye_col = (0, 220, 0) if gaze == "Center" else (0, 165, 255)
    put("EYE CONTACT", (180, 180, 255), 0.5, 1)
    put(f"  Gaze: {gaze}", eye_col)
    put(f"  Contact: {eye_pct}%")
    put("-" * 28, (80, 80, 80))

    direction = head_result.get("direction",       "N/A")
    stability = head_result.get("stability_score", 0)
    head_col  = (0, 220, 0) if direction == "Forward" else (0, 165, 255)
    put("HEAD POSE", (180, 180, 255), 0.5, 1)
    put(f"  Dir: {direction}", head_col)
    put(f"  Stability: {stability}/100")
    pitch = head_result.get("pitch", 0)
    yaw   = head_result.get("yaw",   0)
    put(f"  P:{pitch:.1f} Y:{yaw:.1f}")
    put("-" * 28, (80, 80, 80))

    put("SCORE TREND", (180, 180, 255), 0.5, 1)
    if len(score_history) > 1:
        pts = list(score_history)
        gx, gy, gw, gh = x, y, 230, 50
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (40, 40, 40), -1)
        for i in range(1, len(pts)):
            x1 = gx + int((i - 1) / (SCORE_HISTORY_LEN - 1) * gw)
            x2 = gx + int(i       / (SCORE_HISTORY_LEN - 1) * gw)
            y1 = gy + gh - int(pts[i - 1] / 100 * gh)
            y2 = gy + gh - int(pts[i]     / 100 * gh)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 100), 1)
        y += gh + 8

    put("-" * 28, (80, 80, 80))
    put("TIPS", (180, 180, 255), 0.5, 1)
    if gaze != "Center":
        put("  Look at the camera",  (0, 200, 255), 0.45)
    if nerv > 50:
        put("  Breathe, slow down",  (0, 200, 255), 0.45)
    if direction != "Forward":
        put("  Face forward",        (0, 200, 255), 0.45)
    if nerv <= 50 and gaze == "Center" and direction == "Forward":
        put("  Great job! Keep it up", (0, 220, 0), 0.45)

    # ── Interview status overlay (bottom of frame) ──
    if current_question:
        words     = current_question.split()
        max_chars = 55
        line1     = ""
        line2     = ""
        for word in words:
            if len(line1) + len(word) + 1 <= max_chars:
                line1 += (" " if line1 else "") + word
            else:
                line2 += (" " if line2 else "") + word

        cv2.rectangle(frame, (0, h - 75), (panel_x - 20, h), (20, 20, 20), -1)
        cv2.putText(frame, "Q: " + line1, (10, h - 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        if line2:
            cv2.putText(frame, "   " + line2, (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

    if status:
        cv2.putText(frame, status, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    return frame


# SESSION SUMMARY

def print_summary(score_history, expr_counts, total_frames, duration_sec,
                  interview_results=None):
    if not score_history:
        print("\n[INFO] No data recorded.")
        return

    avg_score = int(np.mean(score_history))
    max_score = int(np.max(score_history))
    min_score = int(np.min(score_history))
    label, _  = confidence_label(avg_score)

    print("\n" + "=" * 50)
    print("  SESSION SUMMARY")
    print("=" * 50)
    print(f"  Duration        : {duration_sec:.1f} seconds")
    print(f"  Frames Analyzed : {total_frames}")
    print(f"  Avg Score       : {avg_score}/100  [{label}]")
    print(f"  Peak Score      : {max_score}/100")
    print(f"  Lowest Score    : {min_score}/100")
    print("\n  Expression Breakdown:")
    total_expr = sum(expr_counts.values()) or 1
    for expr, count in sorted(expr_counts.items(), key=lambda x: -x[1]):
        pct = int(count / total_expr * 100)
        print(f"    {expr:<16} {pct:>3}%  {'#' * (pct // 5)}")

    if interview_results:
        print("\n" + "=" * 50)
        print("  INTERVIEW Q&A SUMMARY")
        print("=" * 50)
        for i, r in enumerate(interview_results, 1):
            fb = r.get('feedback', {})
            if isinstance(fb, dict):
                score_str  = fb.get('score_str', '?')
                strength   = fb.get('strength', '')
                improve    = fb.get('improvement', '')
                vis_conf   = fb.get('visual_confidence', '?')
            else:
                score_str  = str(fb)
                strength   = improve = vis_conf = ''

            print(f"\n  Q{i}: {r['question']}")
            print(f"  Answer   : {r['answer'][:200]}")
            if r.get('followup'):
                print(f"  Follow-up: {r['followup']}")
                print(f"  FU Answer: {r.get('followup_answer','')[:120]}")
            print(f"  AI Score : {score_str}  |  Visual Conf: {vis_conf}/100")
            print(f"  Strength : {strength}")
            print(f"  Improve  : {improve}")
            print("  " + "-" * 46)

        print("\n  OVERALL AI FEEDBACK:")
        try:
            last_role = interview_results[-1].get('job_role', 'the role')
            summary   = generate_final_summary(interview_results, last_role)
            print(f"  Overall Score      : {summary.get('overall_score_str','')}")
            print(f"  Top Strength       : {summary.get('top_strength','')}")
            print(f"  Top Improvement    : {summary.get('top_area_to_improve','')}")
            print(f"  Weak Topics        : {', '.join(summary.get('weak_topics',[]))}")
            print(f"  Final Tip          : {summary.get('final_tip','')}")
        except Exception as e:
            print(f"  [WARN] Summary generation failed: {e}")

    print("\n  Improvement Tips:")
    if avg_score < 40:
        print("    - Practice maintaining eye contact")
        print("    - Work on reducing visible nervousness")
        print("    - Keep your head stable and facing forward")
    elif avg_score < 70:
        print("    - Good effort — focus on eye contact consistency")
        print("    - Try to appear more relaxed")
    else:
        print("    - Strong performance!")
        print("    - Keep practicing to maintain consistency")
    print("=" * 50)



# COLLECT USER PROFILE


def collect_profile() -> dict:
    """
    Returns a full profile dict with keys:
    name, job_role, experience, skills, projects, education, summary, resume_text
    """
    print("\n" + "=" * 50)
    print("  INTERVIEW SETUP")
    print("=" * 50)
    print("  [1]  Upload Resume (PDF / DOCX / TXT / MD) — auto fill")
    print("  [2]  Enter manually")
    print("=" * 50)
    choice = input("  Choice: ").strip()

    if choice == "1":
        path = input("  Resume path: ").strip().strip('"')
        print("  Parsing resume...")
        try:
            profile = resume_to_profile(path)
            print(f"\n  Parsed Profile:")
            print(f"  Name       : {profile['name']}")
            print(f"  Role       : {profile['job_role']}")
            print(f"  Experience : {profile['experience']}")
            print(f"  Skills     : {profile['skills']}")
            if profile.get('projects'):
                print(f"  Projects   : {', '.join(profile['projects'][:3])}")
            if profile.get('education'):
                print(f"  Education  : {profile['education']}")
            confirm = input("\n  Looks good? (y/n): ").strip().lower()
            if confirm == 'y':
                return profile
        except Exception as e:
            print(f"  [WARN] Resume parse failed: {e}. Falling back to manual entry.")

    # manual fallback — build the same dict shape
    name       = input("  Your Name       : ").strip()
    job_role   = input("  Job Role        : ").strip()
    experience = input("  Experience      : ").strip()
    skills     = input("  Skills          : ").strip()
    print("\n  Resume text (paste, press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return {
        'name':        name,
        'job_role':    job_role,
        'experience':  experience,
        'skills':      skills,
        'projects':    [],
        'education':   '',
        'summary':     '',
        'resume_text': '\n'.join(lines),
    }


# MAIN — INTERVIEW LIVE SESSION


def run_live_session():
    print("\n" + "=" * 50)
    print("  AI INTERVIEW CONFIDENCE ANALYZER")
    print("=" * 50)
    print("  Press  Q  to quit and see summary")
    print("  Press  S  to take a snapshot")
    print("=" * 50 + "\n")

    profile    = collect_profile()
    name       = profile['name']
    job_role   = profile['job_role']
    experience = profile['experience']
    skills     = profile['skills']
    resume_ctx = get_resume_context_for_llm(profile)   # rich context string

    print("\n[INFO] Generating interview questions...")
    questions = generate_questions(
        name, job_role, experience, skills,
        resume_text=resume_ctx,
        num_questions=2,
    )
    print(f"[INFO] {len(questions)} questions ready.\n")

    landmark_extractor = FaceLandmarkExtractor()
    if not getattr(landmark_extractor, 'enabled', False):
        print('[ERROR] FaceLandmarkExtractor not enabled. Check MediaPipe.')
        return
    expr_detector = ExpressionDetector(fps=30)
    eye_detector  = EyeContactDetector()
    head_estimator = HeadPoseEstimator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam."); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    score_history    = deque(maxlen=SCORE_HISTORY_LEN)
    all_scores       = []
    expr_counts      = {}
    total_frames     = 0
    start_time       = time.time()
    snapshot_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(snapshot_dir, exist_ok=True)

    empty_expr = {"expression": "N/A", "nervousness_score": 0,
                  "expression_score": 0, "blink_rate": 0}
    empty_eye  = {"gaze_direction": "N/A", "eye_contact_pct": 0, "score": 0}
    empty_head = {"direction": "N/A", "stability_score": 0, "pitch": 0.0, "yaw": 0.0}

    # ── Interview state ──
    interview_results  = []
    current_q_idx      = 0
    current_question   = questions[current_q_idx]
    status_text        = "Listening... (speak your answer)"
    answer_scores      = []    # confidence scores during this answer
    answer_start_time  = time.time()
    ANSWER_DURATION    = 45    # seconds per answer

    # Speak first question in background thread
    threading.Thread(
        target=speak,
        args=(f"Welcome {name}. Question 1. {current_question}",),
        daemon=True
    ).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam frame."); break

        total_frames += 1
        lm_result = landmark_extractor.extract(frame)
        disp      = lm_result["annotated_frame"].copy()

        if lm_result["face_detected"]:
            kp        = lm_result["key_points"]
            landmarks = lm_result["landmarks"]

            expr_result = expr_detector.detect(kp, frame.shape)
            eye_result  = eye_detector.detect(kp, frame.shape)
            head_result = head_estimator.detect(landmarks, frame.shape)
            confidence  = compute_confidence_score(expr_result, eye_result, head_result)

            score_history.append(confidence)
            all_scores.append(confidence)
            answer_scores.append(confidence)

            expr = expr_result.get("expression", "N/A")
            expr_counts[expr] = expr_counts.get(expr, 0) + 1

            disp = draw_dashboard(disp, expr_result, eye_result, head_result,
                                  confidence, score_history,
                                  current_question, status_text)
        else:
            cv2.putText(disp, "No face detected — position yourself in frame",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
            disp = draw_dashboard(disp, empty_expr, empty_eye, empty_head,
                                  0, score_history, current_question, status_text)

        # ── Timer per answer ──
        elapsed_answer = time.time() - answer_start_time
        remaining      = max(0, int(ANSWER_DURATION - elapsed_answer))
        cv2.putText(disp, f"Answer time left: {remaining}s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

        # ── Move to next question when time is up ──
        if elapsed_answer >= ANSWER_DURATION and current_q_idx < len(questions):
            status_text = "Transcribing your answer..."
            cv2.imshow("AI Interview Analyzer  (Q=quit  S=snapshot)", disp)
            cv2.waitKey(1)

            # ── 1. Transcribe answer via STT ──────────────────────────────────
            avg_conf = int(np.mean(answer_scores)) if answer_scores else 0
            try:
                transcribed_answer = transcribe()   # records mic and returns text
            except Exception as e:
                print(f"[WARN] STT failed: {e}")
                transcribed_answer = f"[STT unavailable — visual confidence: {avg_conf}/100]"

            status_text = "Evaluating with AI..."
            cv2.imshow("AI Interview Analyzer  (Q=quit  S=snapshot)", disp)
            cv2.waitKey(1)

            # ── 2. Check if a follow-up question is warranted ─────────────────
            followup_q      = None
            followup_answer = ""
            if transcribed_answer and not transcribed_answer.startswith('['):
                try:
                    followup_q = generate_followup(current_question, transcribed_answer, job_role)
                except Exception as e:
                    print(f"[WARN] Follow-up generation failed: {e}")

            if followup_q:
                status_text = f"Follow-up: {followup_q[:60]}..."
                threading.Thread(
                    target=speak,
                    args=(f"Follow-up: {followup_q}",),
                    daemon=True
                ).start()
                # Give candidate FOLLOW_UP_DURATION seconds to answer follow-up
                FOLLOW_UP_DURATION = 30
                fu_start = time.time()
                while time.time() - fu_start < FOLLOW_UP_DURATION:
                    ret2, frame2 = cap.read()
                    if ret2:
                        fu_remaining = max(0, int(FOLLOW_UP_DURATION - (time.time() - fu_start)))
                        cv2.putText(frame2,
                                    f"Follow-up time left: {fu_remaining}s",
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
                        cv2.putText(frame2,
                                    f"Follow-up: {followup_q[:70]}",
                                    (10, frame2.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
                        cv2.imshow("AI Interview Analyzer  (Q=quit  S=snapshot)", frame2)
                        cv2.waitKey(1)
                try:
                    followup_answer = transcribe()
                except Exception:
                    followup_answer = ""

            # ── 3. LLM evaluation ────────────────────────────────────────────
            try:
                feedback = evaluate_answer(
                    question=current_question,
                    answer=transcribed_answer,
                    job_role=job_role,
                    followup=followup_q or '',
                    followup_answer=followup_answer,
                )
            except Exception as e:
                print(f"[WARN] Evaluation failed: {e}")
                feedback = {
                    'score': avg_conf // 10,
                    'score_str': f"{avg_conf // 10}/10",
                    'strength': 'Answer recorded.',
                    'improvement': 'AI evaluation unavailable.',
                    'detail': '',
                    'raw': '',
                }

            # Append visual confidence as extra context
            feedback['visual_confidence'] = avg_conf

            interview_results.append({
                'question':        current_question,
                'answer':          transcribed_answer,
                'followup':        followup_q,
                'followup_answer': followup_answer,
                'feedback':        feedback,
                'job_role':        job_role,
            })

            # Print quick feedback to terminal
            print(f"\n[EVAL] Q: {current_question}")
            print(f"[EVAL] A: {transcribed_answer[:120]}...")
            print(f"[EVAL] Score: {feedback.get('score_str','?')}  |  "
                  f"{feedback.get('strength','')}")

            answer_scores     = []
            current_q_idx    += 1
            answer_start_time = time.time()

            if current_q_idx < len(questions):
                current_question = questions[current_q_idx]
                status_text      = "Listening... (speak your answer)"
                threading.Thread(
                    target=speak,
                    args=(f"Question {current_q_idx + 1}. {current_question}",),
                    daemon=True
                ).start()
            else:
                status_text = "Interview complete! Press Q to see summary."
                threading.Thread(
                    target=speak,
                    args=("Interview complete. Great job! Press Q to see your summary.",),
                    daemon=True
                ).start()

        # ── Elapsed session timer ──
        elapsed = int(time.time() - start_time)
        cv2.putText(disp, f"{elapsed//60:02d}:{elapsed%60:02d}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("AI Interview Analyzer  (Q=quit  S=snapshot)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(snapshot_dir, f"snapshot_{ts}.jpg")
            cv2.imwrite(path, disp)
            print(f"[INFO] Snapshot saved: {path}")

    cap.release()
    landmark_extractor.release()
    cv2.destroyAllWindows()

    duration = time.time() - start_time
    print_summary(all_scores, expr_counts, total_frames, duration, interview_results)



# MAIN — IMAGE TEST (unchanged)


def run_image_test(image_path):
    print(f"\n[INFO] Running on image: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot load: {image_path}"); return

    landmark_extractor = FaceLandmarkExtractor()
    expr_detector      = ExpressionDetector(fps=30)
    eye_detector       = EyeContactDetector()
    head_estimator     = HeadPoseEstimator()

    lm_result = landmark_extractor.extract_image(frame)

    if not lm_result["face_detected"]:
        print("[ERROR] No face detected in image."); return

    kp        = lm_result["key_points"]
    landmarks = lm_result["landmarks"]

    expr_result = expr_detector.detect(kp, frame.shape)
    eye_result  = eye_detector.detect(kp, frame.shape)
    head_result = head_estimator.detect(landmarks, frame.shape)
    confidence  = compute_confidence_score(expr_result, eye_result, head_result)
    label, _    = confidence_label(confidence)

    print("\n" + "=" * 50)
    print("  ANALYSIS RESULT")
    print("=" * 50)
    print(f"  Confidence Score : {confidence}/100  [{label}]")
    print(f"  Expression       : {expr_result['expression']}")
    print(f"  Nervousness      : {expr_result['nervousness_score']}/100")
    print(f"  Gaze             : {eye_result['gaze_direction']}")
    print(f"  Eye Contact      : {eye_result['eye_contact_pct']}%")
    print(f"  Head Direction   : {head_result['direction']}")
    print(f"  Head Stability   : {head_result['stability_score']}/100")
    print("=" * 50)

    disp = draw_dashboard(
        lm_result["annotated_frame"].copy(),
        expr_result, eye_result, head_result,
        confidence, deque([confidence])
    )
    cv2.imshow("AI Interview Analyzer - Image Test (any key to close)", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    landmark_extractor.release()



# ENTRY POINT


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--image":
        run_image_test(sys.argv[2]); sys.exit(0)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--live":
        run_live_session(); sys.exit(0)

    print("\n" + "=" * 50)
    print("  AI INTERVIEW CONFIDENCE ANALYZER")
    print("=" * 50)
    print("  [1]  Live interview session (with AI questions)")
    print("  [2]  Test on image")
    print("=" * 50)
    choice = input("  Choice (1 or 2): ").strip()

    if choice == "1":
        run_live_session()
    elif choice == "2":
        path = input("  Image path: ").strip().strip('"')
        run_image_test(path)
    else:
        print("  Invalid choice.")