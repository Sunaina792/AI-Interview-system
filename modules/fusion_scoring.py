# MODULE 5: Fusion Scoring
# AI Interview Confidence & Behavior Analysis System


import cv2
import numpy as np
from collections import deque


# CONFIGURATION

SCORE_HISTORY_WINDOW = 30       # frames for rolling average

WEIGHTS = {
    "eye_contact":  0.30,
    "expression":   0.25,
    "head_pose":    0.25,
    "audio":        0.20,
}

COACHING_TIPS = {
    "eye_contact": {
        "low":  "Maintain eye contact — look into the camera, not at the screen edges",
        "mid":  "Good gaze direction — try to stay consistent",
        "high": "Excellent eye contact",
    },
    "expression": {
        "low":  "Relax your face — tension around the mouth and brows signals nervousness",
        "mid":  "Expression is mostly calm — slight improvements possible",
        "high": "Facial expression looks confident and composed",
    },
    "head_pose": {
        "low":  "Keep your head steady and centered — avoid excessive nodding or tilting",
        "mid":  "Head pose is acceptable — try to reduce side tilts",
        "high": "Head position looks stable and professional",
    },
    "audio": {
        "low":  "Speak clearly and steadily — avoid long pauses and filler words",
        "mid":  "Voice is decent — work on projection and pace",
        "high": "Voice sounds confident and well-paced",
    },
}


# FUSION SCORER

class FusionScorer:
    def __init__(self):
        self.history = deque(maxlen=SCORE_HISTORY_WINDOW)
        self.session_scores = []
        self.frame_count = 0

    def compute(
        self,
        eye_contact_score: float = 0,
        expression_score:  float = 0,
        head_pose_score:   float = 0,
        audio_score:       float = 0,
    ) -> dict:
        """
        Fuses individual module scores into a single confidence score.

        Each input is expected 0–100.
        Returns dict with fused score, label, breakdown, and tips.
        """
        self.frame_count += 1

        scores = {
            "eye_contact": float(eye_contact_score),
            "expression":  float(expression_score),
            "head_pose":   float(head_pose_score),
            "audio":       float(audio_score),
        }

        raw = sum(scores[k] * WEIGHTS[k] for k in scores)
        self.history.append(raw)
        self.session_scores.append(raw)

        smoothed = round(float(np.mean(self.history)), 1)
        label    = self._label(smoothed)
        tips     = self._tips(scores)
        dominant = self._weakest_signal(scores)

        return {
            "score":          smoothed,
            "raw_score":      round(raw, 1),
            "label":          label,
            "breakdown":      {k: round(scores[k], 1) for k in scores},
            "weights":        WEIGHTS,
            "tips":           tips,
            "weakest_signal": dominant,
            "frame":          self.frame_count,
        }

    def _label(self, score: float) -> str:
        if score >= 75:
            return "Confident"
        elif score >= 50:
            return "Moderate"
        else:
            return "Needs Improvement"

    def _tips(self, scores: dict) -> list:
        tips = []
        for signal, val in scores.items():
            bucket = self._bucket(val)
            tips.append(COACHING_TIPS[signal][bucket])
        return tips

    def _bucket(self, val: float) -> str:
        if val >= 70:
            return "high"
        elif val >= 40:
            return "mid"
        else:
            return "low"

    def _weakest_signal(self, scores: dict) -> str:
        return min(scores, key=lambda k: scores[k])

    def session_summary(self) -> dict:
        """Call at end of session to get overall stats."""
        if not self.session_scores:
            return {}

        arr = np.array(self.session_scores)
        return {
            "avg_score":   round(float(np.mean(arr)), 1),
            "max_score":   round(float(np.max(arr)),  1),
            "min_score":   round(float(np.min(arr)),  1),
            "std_dev":     round(float(np.std(arr)),  1),
            "total_frames": self.frame_count,
            "label":        self._label(float(np.mean(arr))),
        }

    def reset(self):
        self.history.clear()
        self.session_scores.clear()
        self.frame_count = 0


# DRAW OVERLAY

def draw_fusion_overlay(frame, result: dict) -> object:
    score    = result["score"]
    label    = result["label"]
    breakdown = result["breakdown"]
    tip      = result["tips"][0] if result["tips"] else ""
    weakest  = result["weakest_signal"]

    label_color = {
        "Confident":        (0, 220, 0),
        "Moderate":         (0, 200, 255),
        "Needs Improvement":(0, 80, 255),
    }.get(label, (200, 200, 200))

    # Main score bar background
    cv2.rectangle(frame, (8, 8), (350, 105), (30, 30, 30), -1)
    cv2.rectangle(frame, (8, 8), (350, 105), (80, 80, 80), 1)

    cv2.putText(frame, f"Confidence: {score}/100", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2)
    cv2.putText(frame, label, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 1)

    # Score bar fill
    bar_x, bar_y, bar_w, bar_h = 15, 68, 320, 10
    filled = int(bar_w * score / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), label_color, -1)

    # Breakdown panel (bottom-left)
    signals = [
        ("Eye",  breakdown.get("eye_contact", 0)),
        ("Expr", breakdown.get("expression",  0)),
        ("Head", breakdown.get("head_pose",   0)),
        ("Audio",breakdown.get("audio",       0)),
    ]
    panel_y = frame.shape[0] - 90
    cv2.rectangle(frame, (8, panel_y - 15), (260, frame.shape[0] - 8), (30, 30, 30), -1)
    cv2.rectangle(frame, (8, panel_y - 15), (260, frame.shape[0] - 8), (70, 70, 70), 1)

    for i, (name, val) in enumerate(signals):
        x = 15 + i * 62
        color = (0, 200, 0) if val >= 70 else (0, 200, 255) if val >= 40 else (0, 80, 255)
        cv2.putText(frame, name, (x, panel_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        cv2.putText(frame, str(int(val)), (x + 5, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        mini_filled = int(40 * val / 100)
        cv2.rectangle(frame, (x, panel_y + 32), (x + 40, panel_y + 38), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, panel_y + 32), (x + mini_filled, panel_y + 38), color, -1)

    # Coaching tip (top-right area)
    tip_short = tip[:60] + ("..." if len(tip) > 60 else "")
    cv2.putText(frame, f"Tip [{weakest}]: {tip_short}", (10, frame.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 100), 1)

    return frame


# TEST: IMAGE

def test_on_image(image_path: str):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_landmarks     import FaceLandmarkExtractor
    from eye_contact        import EyeContactDetector
    from expression_detection import ExpressionDetector
    from head_pose          import HeadPoseEstimator

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot load: {image_path}")
        return

    extractor  = FaceLandmarkExtractor()
    lm_result  = extractor.extract_image(frame)

    if not lm_result["face_detected"]:
        print("[ERROR] No face detected.")
        return

    kp    = lm_result["key_points"]
    ear   = lm_result["ear"]

    eye_detector  = EyeContactDetector()
    expr_detector = ExpressionDetector()
    head_estimator = HeadPoseEstimator()
    scorer        = FusionScorer()

    eye_result  = eye_detector.detect(kp, frame.shape)
    expr_result = expr_detector.detect(kp, ear)
    head_result = head_estimator.estimate(kp, frame.shape)

    result = scorer.compute(
        eye_contact_score = eye_result.get("score", 0),
        expression_score  = expr_result.get("score", 0),
        head_pose_score   = head_result.get("score", 0),
        audio_score       = 0,
    )

    print("\n" + "="*45)
    print("  MODULE 5 — FUSION SCORE (IMAGE)")
    print("="*45)
    print(f"  Fused Score  : {result['score']}/100")
    print(f"  Label        : {result['label']}")
    print(f"  Eye Contact  : {result['breakdown']['eye_contact']}/100")
    print(f"  Expression   : {result['breakdown']['expression']}/100")
    print(f"  Head Pose    : {result['breakdown']['head_pose']}/100")
    print(f"  Audio        : {result['breakdown']['audio']}/100  (N/A for image)")
    print(f"  Weakest      : {result['weakest_signal']}")
    print("\n  Tips:")
    for tip in result["tips"]:
        print(f"    • {tip}")

    out = lm_result["annotated_frame"].copy()
    out = draw_fusion_overlay(out, result)
    cv2.imshow("Module 5 - Fusion Score (any key to close)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    extractor.release()


# TEST: WEBCAM

def test_webcam():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_landmarks       import FaceLandmarkExtractor
    from eye_contact          import EyeContactDetector
    from expression_detection import ExpressionDetector
    from head_pose            import HeadPoseEstimator

    extractor      = FaceLandmarkExtractor()
    eye_detector   = EyeContactDetector()
    expr_detector  = ExpressionDetector()
    head_estimator = HeadPoseEstimator()
    scorer         = FusionScorer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Webcam started. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm_result = extractor.extract(frame)
        disp      = lm_result["annotated_frame"].copy()

        if lm_result["face_detected"]:
            kp  = lm_result["key_points"]
            ear = lm_result["ear"]

            eye_result  = eye_detector.detect(kp, frame.shape)
            expr_result = expr_detector.detect(kp, ear)
            head_result = head_estimator.estimate(kp, frame.shape)

            result = scorer.compute(
                eye_contact_score = eye_result.get("score", 0),
                expression_score  = expr_result.get("score", 0),
                head_pose_score   = head_result.get("score", 0),
                audio_score       = 0,
            )

            disp = draw_fusion_overlay(disp, result)
        else:
            cv2.putText(disp, "No face detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Module 5 - Fusion Score (Q to quit)", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    summary = scorer.session_summary()
    print("\n" + "="*45)
    print("  SESSION SUMMARY")
    print("="*45)
    print(f"  Avg Score  : {summary.get('avg_score', 0)}/100")
    print(f"  Max Score  : {summary.get('max_score', 0)}/100")
    print(f"  Min Score  : {summary.get('min_score', 0)}/100")
    print(f"  Std Dev    : {summary.get('std_dev', 0)}")
    print(f"  Frames     : {summary.get('total_frames', 0)}")
    print(f"  Overall    : {summary.get('label', 'N/A')}")

    cap.release()
    extractor.release()
    cv2.destroyAllWindows()


# ENTRY POINT

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "--image":
        test_on_image(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--webcam":
        test_webcam()
        sys.exit(0)

    print("\n" + "="*45)
    print("  MODULE 5 - Fusion Scoring")
    print("="*45)
    print("  [1]  Test on IMAGE")
    print("  [2]  Live WEBCAM")
    print("="*45)
    choice = input("  Choice (1 or 2): ").strip()

    if choice == "1":
        path = input("  Image path: ").strip().strip('"')
        test_on_image(path)
    elif choice == "2":
        test_webcam()
    else:
        print("  Invalid choice.")