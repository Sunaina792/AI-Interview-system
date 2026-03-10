# ============================================================
# MODULE 3: Eye Contact Detection
# ============================================================

import cv2
import numpy as np
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================
GAZE_THRESHOLD        = 0.25   # iris offset ratio — beyond this = looking away
HISTORY_WINDOW        = 150    # frames to track eye contact % over
MIN_EYE_CONTACT_SCORE = 40     # below this % = poor eye contact


# ============================================================
# EYE CONTACT DETECTOR
# ============================================================
class EyeContactDetector:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_WINDOW)

    def detect(self, key_points, frame_shape):
        """
        Estimates gaze direction using iris position relative to eye corners.

        Returns dict:
            looking_at_camera : bool
            gaze_direction    : Left / Right / Up / Down / Center
            eye_contact_pct   : rolling % of frames with eye contact
            left_offset       : (x, y) iris offset ratio for left eye
            right_offset      : (x, y) iris offset ratio for right eye
            score             : 0-100
        """
        left_offset  = self._iris_offset(
            key_points.get("left_eye",  []),
            key_points.get("left_iris", [])
        )
        right_offset = self._iris_offset(
            key_points.get("right_eye",  []),
            key_points.get("right_iris", [])
        )

        gaze_dir     = self._gaze_direction(left_offset, right_offset)
        at_camera    = gaze_dir == "Center"

        self.history.append(1 if at_camera else 0)
        eye_contact_pct = round(sum(self.history) / len(self.history) * 100, 1) if self.history else 0.0
        score           = int(eye_contact_pct)

        return {
            "looking_at_camera": at_camera,
            "gaze_direction":    gaze_dir,
            "eye_contact_pct":   eye_contact_pct,
            "left_offset":       left_offset,
            "right_offset":      right_offset,
            "score":             score,
        }

    def _iris_offset(self, eye_pts, iris_pts):
        """
        Measures how far the iris has shifted from the center of the eye.
        Returns (x_ratio, y_ratio) — values near 0 = centered = looking at camera.
        """
        if len(eye_pts) < 4 or not iris_pts:
            return (0.0, 0.0)

        eye_left   = np.array(eye_pts[0])
        eye_right  = np.array(eye_pts[3])
        eye_top    = np.array(eye_pts[1])
        eye_bottom = np.array(eye_pts[5]) if len(eye_pts) > 5 else np.array(eye_pts[-1])

        eye_center_x = (eye_left[0]  + eye_right[0])  / 2
        eye_center_y = (eye_top[1]   + eye_bottom[1])  / 2
        eye_width    = np.linalg.norm(eye_right - eye_left)
        eye_height   = abs(eye_bottom[1] - eye_top[1])

        iris_x, iris_y = iris_pts[0]

        x_offset = (iris_x - eye_center_x) / eye_width  if eye_width  > 0 else 0.0
        y_offset = (iris_y - eye_center_y) / eye_height if eye_height > 0 else 0.0

        return (round(x_offset, 3), round(y_offset, 3))

    def _gaze_direction(self, left_offset, right_offset):
        # Average offset from both eyes
        avg_x = (left_offset[0] + right_offset[0]) / 2
        avg_y = (left_offset[1] + right_offset[1]) / 2

        if abs(avg_x) < GAZE_THRESHOLD and abs(avg_y) < GAZE_THRESHOLD:
            return "Center"
        if avg_x > GAZE_THRESHOLD:
            return "Right"
        if avg_x < -GAZE_THRESHOLD:
            return "Left"
        if avg_y < -GAZE_THRESHOLD:
            return "Up"
        if avg_y > GAZE_THRESHOLD:
            return "Down"
        return "Center"


# ============================================================
# DRAW OVERLAY
# ============================================================
def draw_eye_contact_overlay(frame, result):
    at_cam  = result["looking_at_camera"]
    gaze    = result["gaze_direction"]
    pct     = result["eye_contact_pct"]
    score   = result["score"]

    color = (0, 255, 0) if at_cam else (0, 100, 255)
    label = f"Gaze: {gaze}  ({'ON camera' if at_cam else 'OFF camera'})"

    cv2.putText(frame, label, (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Eye Contact: {pct}%  |  Score: {score}/100", (10, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # Mini gaze indicator box (top right corner)
    bx, by, bw, bh = frame.shape[1] - 90, 10, 80, 60
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (120, 120, 120), 1)

    cx = bx + bw // 2
    cy = by + bh // 2
    lo = result["left_offset"]
    dot_x = int(cx + lo[0] * 25)
    dot_y = int(cy + lo[1] * 20)
    dot_x = max(bx + 5, min(bx + bw - 5, dot_x))
    dot_y = max(by + 5, min(by + bh - 5, dot_y))
    cv2.circle(frame, (cx, cy), 3, (80, 80, 80), -1)
    cv2.circle(frame, (dot_x, dot_y), 6, color, -1)
    cv2.putText(frame, "gaze", (bx + 22, by + bh + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    return frame


# ============================================================
# TEST: IMAGE
# ============================================================
def test_on_image(image_path: str):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_landmarks import FaceLandmarkExtractor

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot load: {image_path}")
        return

    extractor = FaceLandmarkExtractor()
    lm_result = extractor.extract_image(frame)

    if not lm_result["face_detected"]:
        print("[ERROR] No face detected.")
        return

    detector = EyeContactDetector()
    result   = detector.detect(lm_result["key_points"], frame.shape)

    print("\n" + "="*45)
    print("  MODULE 3 — IMAGE TEST RESULT")
    print("="*45)
    print(f"  Gaze Direction   : {result['gaze_direction']}")
    print(f"  Looking at Camera: {result['looking_at_camera']}")
    print(f"  Eye Contact %    : {result['eye_contact_pct']}%")
    print(f"  Score            : {result['score']}/100")
    print(f"  Left Iris Offset : {result['left_offset']}")
    print(f"  Right Iris Offset: {result['right_offset']}")

    out = lm_result["annotated_frame"].copy()
    out = draw_eye_contact_overlay(out, result)
    cv2.imshow("Module 3 - Eye Contact (any key to close)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    extractor.release()


# ============================================================
# TEST: WEBCAM
# ============================================================
def test_webcam():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_landmarks import FaceLandmarkExtractor

    extractor = FaceLandmarkExtractor()
    detector  = EyeContactDetector()
    cap       = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Webcam started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lm_result = extractor.extract(frame)
        disp      = lm_result["annotated_frame"].copy()

        if lm_result["face_detected"]:
            result = detector.detect(lm_result["key_points"], frame.shape)
            disp   = draw_eye_contact_overlay(disp, result)

        cv2.imshow("Module 3 - Eye Contact (Q to quit)", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    extractor.release()
    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "--image":
        test_on_image(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--webcam":
        test_webcam()
        sys.exit(0)

    print("\n" + "="*45)
    print("  MODULE 3 - Eye Contact Detection")
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