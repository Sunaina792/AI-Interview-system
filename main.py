
# main.py — AI Interview Confidence Analyzer
# Combines: Face Landmarks + Expression + Eye Contact + Head Pose


import cv2
import sys
import os
import time
import numpy as np
from collections import deque

# Add modules folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

from modules.face_landmarks      import FaceLandmarkExtractor
from modules.expression_detection import ExpressionDetector
from modules.eye_contact          import EyeContactDetector
from modules.head_pose            import HeadPoseEstimator


# CONFIDENCE SCORE WEIGHTS (must sum to 100)

WEIGHTS = {
    "eye_contact":   30,
    "expression":    25,
    "head_stability": 25,
    "nervousness":   20,   # inverted — lower nervousness = higher score
}


# SCORE HISTORY for smooth rolling average

SCORE_HISTORY_LEN = 45   # ~1.5 seconds at 30fps



# CONFIDENCE CALCULATOR

def compute_confidence_score(expr_result, eye_result, head_result):
    """
    Combines outputs from all modules into one 0-100 confidence score.

    Like a judge at a talent show with 4 scorecards —
    each module gives a score, we weight and average them.
    """
    eye_score   = eye_result.get("score",             0)
    expr_score  = expr_result.get("expression_score", 0)
    head_score  = head_result.get("stability_score",  0)
    nerv_score  = expr_result.get("nervousness_score", 0)

    # Nervousness is inverted: 0 nervousness = 100 confidence contribution
    nerv_contribution = max(0, 100 - nerv_score)

    final = (
        eye_score          * WEIGHTS["eye_contact"]    / 100 +
        expr_score         * WEIGHTS["expression"]     / 100 +
        head_score         * WEIGHTS["head_stability"] / 100 +
        nerv_contribution  * WEIGHTS["nervousness"]    / 100
    )
    return min(100, max(0, int(final)))


def confidence_label(score):
    if score >= 80: return "High",    (0, 220, 0)
    if score >= 60: return "Moderate",(0, 200, 150)
    if score >= 40: return "Low",     (0, 165, 255)
    return "Very Low", (0, 60, 255)



# DRAW DASHBOARD PANEL (right side overlay)

def draw_dashboard(frame, expr_result, eye_result, head_result, confidence, score_history):
    h, w = frame.shape[:2]

    # Semi-transparent dark panel on the right
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

    # ── Title ──
    put("CONFIDENCE ANALYZER", (255, 255, 255), 0.55, 2)
    put("-" * 28, (80, 80, 80))

    # ── Final Score ──
    label, color = confidence_label(confidence)
    put(f"SCORE: {confidence}/100", color, 0.75, 2)
    put(f"Level: {label}", color)

    # Score bar
    bar_w = int((confidence / 100) * 230)
    cv2.rectangle(frame, (x, y),          (x + 230, y + 12), (60, 60, 60),  -1)
    cv2.rectangle(frame, (x, y),          (x + bar_w, y + 12), color,        -1)
    y += 22
    put("-" * 28, (80, 80, 80))

    # ── Expression ──
    expr      = expr_result.get("expression", "N/A")
    nerv      = expr_result.get("nervousness_score", 0)
    blink_r   = expr_result.get("blink_rate", 0)
    expr_col  = (0, 220, 0) if expr == "Happy" else (0, 165, 255) if nerv > 40 else (220, 220, 220)
    put("EXPRESSION", (180, 180, 255), 0.5, 1)
    put(f"  {expr}", expr_col)
    put(f"  Nerv: {nerv}/100  Blink:{blink_r}/m")

    put("-" * 28, (80, 80, 80))

    # ── Eye Contact ──
    gaze      = eye_result.get("gaze_direction",  "N/A")
    eye_pct   = eye_result.get("eye_contact_pct", 0)
    eye_col   = (0, 220, 0) if gaze == "Center" else (0, 165, 255)
    put("EYE CONTACT", (180, 180, 255), 0.5, 1)
    put(f"  Gaze: {gaze}", eye_col)
    put(f"  Contact: {eye_pct}%")

    put("-" * 28, (80, 80, 80))

    # ── Head Pose ──
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

    # ── Mini score graph ──
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

    # ── Tips ──
    put("TIPS", (180, 180, 255), 0.5, 1)
    if gaze != "Center":
        put("  Look at the camera", (0, 200, 255), 0.45)
    if nerv > 50:
        put("  Breathe, slow down", (0, 200, 255), 0.45)
    if direction != "Forward":
        put("  Face forward", (0, 200, 255), 0.45)
    if nerv <= 50 and gaze == "Center" and direction == "Forward":
        put("  Great job! Keep it up", (0, 220, 0), 0.45)

    return frame



# SESSION SUMMARY

def print_summary(score_history, expr_counts, total_frames, duration_sec):
    if not score_history:
        print("\n[INFO] No data recorded.")
        return

    avg_score = int(np.mean(score_history))
    max_score = int(np.max(score_history))
    min_score = int(np.min(score_history))
    label, _  = confidence_label(avg_score)

    print("\n" + "="*50)
    print("  SESSION SUMMARY")
    print("="*50)
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
    print("="*50)



# MAIN — WEBCAM LIVE SESSION

def run_live_session():
    print("\n" + "="*50)
    print("  AI INTERVIEW CONFIDENCE ANALYZER")
    print("="*50)
    print("  Press  Q  to quit and see summary")
    print("  Press  S  to take a snapshot")
    print("="*50 + "\n")

    # Init all modules
    landmark_extractor = FaceLandmarkExtractor()
    expr_detector      = ExpressionDetector(fps=30)
    eye_detector       = EyeContactDetector()
    head_estimator     = HeadPoseEstimator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam."); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    score_history = deque(maxlen=SCORE_HISTORY_LEN)
    all_scores    = []
    expr_counts   = {}
    total_frames  = 0
    start_time    = time.time()
    snapshot_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(snapshot_dir, exist_ok=True)

    # Default empty results (shown before face detected)
    empty_expr = {"expression": "N/A", "nervousness_score": 0,
                  "expression_score": 0, "blink_rate": 0}
    empty_eye  = {"gaze_direction": "N/A", "eye_contact_pct": 0, "score": 0}
    empty_head = {"direction": "N/A", "stability_score": 0,
                  "pitch": 0.0, "yaw": 0.0}

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

            expr = expr_result.get("expression", "N/A")
            expr_counts[expr] = expr_counts.get(expr, 0) + 1

            disp = draw_dashboard(disp, expr_result, eye_result, head_result,
                                  confidence, score_history)
        else:
            # No face — show waiting message
            cv2.putText(disp, "No face detected — position yourself in frame",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
            disp = draw_dashboard(disp, empty_expr, empty_eye, empty_head,
                                  0, score_history)

        # Elapsed time (top left)
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
    print_summary(all_scores, expr_counts, total_frames, duration)



# MAIN — IMAGE TEST

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

    print("\n" + "="*50)
    print("  ANALYSIS RESULT")
    print("="*50)
    print(f"  Confidence Score : {confidence}/100  [{label}]")
    print(f"  Expression       : {expr_result['expression']}")
    print(f"  Nervousness      : {expr_result['nervousness_score']}/100")
    print(f"  Gaze             : {eye_result['gaze_direction']}")
    print(f"  Eye Contact      : {eye_result['eye_contact_pct']}%")
    print(f"  Head Direction   : {head_result['direction']}")
    print(f"  Head Stability   : {head_result['stability_score']}/100")
    print("="*50)

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

    print("\n" + "="*50)
    print("  AI INTERVIEW CONFIDENCE ANALYZER")
    print("="*50)
    print("  [1]  Live webcam session")
    print("  [2]  Test on image")
    print("="*50)
    choice = input("  Choice (1 or 2): ").strip()

    if choice == "1":
        run_live_session()
    elif choice == "2":
        path = input("  Image path: ").strip().strip('"')
        run_image_test(path)
    else:
        print("  Invalid choice.")