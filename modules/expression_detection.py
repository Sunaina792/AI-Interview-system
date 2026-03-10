
# MODULE 2: Facial Expression & Nervousness Detection


import cv2
import numpy as np
from collections import deque, Counter

BLINK_THRESHOLD         = 0.16
BLINK_CONSEC_FRAMES     = 4    # eye must stay closed 3 frames to count (was 2)
BLINK_RATE_WINDOW       = 450    # wider window = smoother blink rate (was 300)
HIGH_BLINK_RATE         = 30     # raised threshold — 30+/min = nervous (was 25)
SMILE_THRESHOLD         = 0.28
LIP_COMPRESS_THRESHOLD  = 0.018
EYEBROW_RAISE_THRESHOLD = 0.09   # raised slightly to reduce false positives


class ExpressionDetector:
    def __init__(self, fps=30):
        self.fps                = fps
        self.blink_counter      = 0
        self.total_blinks       = 0
        self.blink_history      = deque(maxlen=BLINK_RATE_WINDOW)
        self.expression_history = deque(maxlen=20)   # wider smooth window (was 10)

    def detect(self, key_points, frame_shape):
        h, w = frame_shape[:2]

        smile          = self._detect_smile(key_points, w)
        lip_compressed = self._detect_lip_compression(key_points, h)
        eyebrow_raised = self._detect_eyebrow_raise(key_points, h)
        ear_avg        = self._get_ear(key_points)
        blink_rate     = self._update_blink(ear_avg)
        nervousness    = self._compute_nervousness(blink_rate, lip_compressed, eyebrow_raised, ear_avg)
        expression, score = self._classify_expression(smile, lip_compressed, eyebrow_raised, nervousness)

        self.expression_history.append(expression)

        return {
            "expression":        self._smooth_expression(),
            "expression_score":  score,
            "smile":             smile,
            "lip_compressed":    lip_compressed,
            "eyebrow_raised":    eyebrow_raised,
            "blink_rate":        blink_rate,
            "nervousness_score": nervousness,
            "total_blinks":      self.total_blinks,
        }

    def _get_ear(self, key_points):
        def ear(pts):
            if len(pts) < 6:
                return 0.3
            A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
            B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
            return (A + B) / (2.0 * C) if C != 0 else 0.3
        l = ear(key_points.get("left_eye",  []))
        r = ear(key_points.get("right_eye", []))
        return (l + r) / 2.0

    def _update_blink(self, ear_avg):
        if ear_avg < BLINK_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= BLINK_CONSEC_FRAMES:
                self.total_blinks += 1
                self.blink_history.append(1)
            else:
                self.blink_history.append(0)
            self.blink_counter = 0

        if len(self.blink_history) > 0:
            blinks_in_window = sum(self.blink_history)
            minutes = len(self.blink_history) / (self.fps * 60)
            return round(blinks_in_window / minutes, 1) if minutes > 0 else 0.0
        return 0.0

    def _detect_smile(self, key_points, frame_width):
        mouth = key_points.get("mouth", [])
        if len(mouth) < 4:
            return False
        mouth_width = np.linalg.norm(np.array(mouth[2]) - np.array(mouth[3]))
        return (mouth_width / frame_width) > SMILE_THRESHOLD

    def _detect_lip_compression(self, key_points, frame_height):
        mouth = key_points.get("mouth", [])
        if len(mouth) < 2:
            return False
        lip_height = abs(np.array(mouth[1])[1] - np.array(mouth[0])[1])
        return (lip_height / frame_height) < LIP_COMPRESS_THRESHOLD

    def _detect_eyebrow_raise(self, key_points, frame_height):
        left_brow = key_points.get("left_eyebrow", [])
        left_eye  = key_points.get("left_eye",     [])
        if not left_brow or not left_eye:
            return False
        brow_y = np.mean([p[1] for p in left_brow])
        eye_y  = np.mean([p[1] for p in left_eye])
        return (abs(eye_y - brow_y) / frame_height) > EYEBROW_RAISE_THRESHOLD

    def _compute_nervousness(self, blink_rate, lip_compressed, eyebrow_raised, ear_avg):
        score = 0

        # Blink rate scoring — tighter thresholds
        if blink_rate > HIGH_BLINK_RATE:
            score += 25
        elif blink_rate > 22:
            score += 15
        elif blink_rate > 16:
            score += 5

        if lip_compressed:
            score += 25
        if eyebrow_raised:
            score += 15    # reduced from 20 — eyebrow raise is less reliable

        # Squinting only counts if sustained (ear in narrow band)
        if 0.13 < ear_avg < 0.20:
            score += 15    # reduced from 20

        return min(score, 100)

    def _classify_expression(self, smile, lip_compressed, eyebrow_raised, nervousness):
        if smile and nervousness < 35:
            return "Happy", 85
        if nervousness >= 65:
            return "Stressed", 80
        if nervousness >= 40:
            return "Nervous", 75
        if lip_compressed:
            return "Tense", 70
        return "Neutral", 80

    def _smooth_expression(self):
        if not self.expression_history:
            return "Neutral"
        return Counter(self.expression_history).most_common(1)[0][0]


def draw_expression_overlay(frame, result):
    expr    = result["expression"]
    nerv    = result["nervousness_score"]
    blink_r = result["blink_rate"]
    total_b = result["total_blinks"]

    colors = {
        "Happy":   (0, 255, 0),   "Neutral": (255, 255, 255),
        "Nervous": (0, 165, 255), "Tense":   (0, 100, 255),
        "Stressed":(0, 0, 255),
    }
    cv2.putText(frame, f"Expression : {expr}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors.get(expr, (255,255,255)), 2)
    cv2.putText(frame, f"Nervousness: {nerv}/100", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255) if nerv > 50 else (200,200,200), 1)
    cv2.putText(frame, f"Blink Rate : {blink_r}/min  (Total: {total_b})", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    flags = []
    if result["smile"]:          flags.append("Smiling")
    if result["lip_compressed"]: flags.append("Lips Tight")
    if result["eyebrow_raised"]: flags.append("Brows Up")
    if flags:
        cv2.putText(frame, " | ".join(flags), (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,100), 1)
    return frame


def test_on_image(image_path: str):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_landmarks import FaceLandmarkExtractor
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot load: {image_path}"); return
    extractor = FaceLandmarkExtractor()
    lm_result = extractor.extract_image(frame)
    if not lm_result["face_detected"]:
        print("[ERROR] No face detected."); return
    detector = ExpressionDetector(fps=30)
    result   = detector.detect(lm_result["key_points"], frame.shape)
    print("\n" + "="*45)
    print("  MODULE 2 - IMAGE TEST RESULT")
    print("="*45)
    for k in ["expression","expression_score","nervousness_score","smile","lip_compressed","blink_rate"]:
        print(f"  {k:<20}: {result[k]}")
    out = lm_result["annotated_frame"].copy()
    out = draw_expression_overlay(out, result)
    cv2.imshow("Module 2 - Expression (any key to close)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    extractor.release()


def test_webcam():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_landmarks import FaceLandmarkExtractor
    extractor = FaceLandmarkExtractor()
    detector  = ExpressionDetector(fps=30)
    cap       = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam."); return
    print("[INFO] Webcam started. Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        lm_result = extractor.extract(frame)
        disp      = lm_result["annotated_frame"].copy()
        if lm_result["face_detected"]:
            result = detector.detect(lm_result["key_points"], frame.shape)
            disp   = draw_expression_overlay(disp, result)
        cv2.imshow("Module 2 - Expression (Q to quit)", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    extractor.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--image":
        test_on_image(sys.argv[2]); sys.exit(0)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--webcam":
        test_webcam(); sys.exit(0)
    print("\n" + "="*45)
    print("  MODULE 2 - Expression Detection")
    print("="*45)
    print("  [1]  Test on IMAGE\n  [2]  Live WEBCAM")
    print("="*45)
    choice = input("  Choice (1 or 2): ").strip()
    if choice == "1":
        test_on_image(input("  Image path: ").strip().strip('"'))
    elif choice == "2":
        test_webcam()
    else:
        print("  Invalid choice.")