# ============================================================
# MODULE 4: Head Pose Estimation
# ============================================================

import cv2
import numpy as np
from collections import deque

PITCH_THRESHOLD  = 20    # raised — compensates for low camera angle
YAW_THRESHOLD    = 28
PITCH_OFFSET     = 8.0    # raised — reduces false Left/Right (was 20)
ROLL_THRESHOLD   = 15
STABILITY_WINDOW = 90    # wider window = smoother stability score (was 60)

FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -330.0, -65.0  ],
    [-225.0, 170.0, -135.0 ],
    [225.0,  170.0, -135.0 ],
    [-150.0,-150.0, -125.0 ],
    [150.0, -150.0, -125.0 ],
], dtype=np.float64)

FACE_2D_INDICES = [1, 152, 33, 263, 78, 308]


class HeadPoseEstimator:
    def __init__(self):
        self.pitch_history = deque(maxlen=STABILITY_WINDOW)
        self.yaw_history   = deque(maxlen=STABILITY_WINDOW)
        self.roll_history  = deque(maxlen=STABILITY_WINDOW)

    def detect(self, landmarks, frame_shape):
        if landmarks is None:
            return self._empty_result()

        h, w    = frame_shape[:2]
        face_2d = self._get_2d_points(landmarks, w, h)
        if face_2d is None:
            return self._empty_result()

        pitch, yaw, roll = self._solve_pnp(face_2d, w, h)

        self.pitch_history.append(abs(pitch))
        self.yaw_history.append(abs(yaw))
        self.roll_history.append(abs(roll))

        direction       = self._get_direction(pitch, yaw, roll)
        stability_score = self._compute_stability()

        return {
            "pitch":           round(pitch, 2),
            "yaw":             round(yaw,   2),
            "roll":            round(roll,  2),
            "direction":       direction,
            "stability_score": stability_score,
            "is_stable":       stability_score >= 60,
        }

    def _get_2d_points(self, landmarks, w, h):
        try:
            pts     = []
            lm_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
            for idx in FACE_2D_INDICES:
                lm = lm_list[idx]
                pts.append([lm.x * w, lm.y * h])
            return np.array(pts, dtype=np.float64)
        except Exception:
            return None

    def _solve_pnp(self, face_2d, w, h):
        focal_length = w
        cam_matrix   = np.array([
            [focal_length, 0,            w / 2],
            [0,            focal_length, h / 2],
            [0,            0,            1    ]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, _ = cv2.solvePnP(
            FACE_3D_POINTS, face_2d, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rot_mat, _         = cv2.Rodrigues(rot_vec)
        angles, _,_,_,_,_ = cv2.RQDecomp3x3(rot_mat)

        pitch = angles[0] * 360
        yaw   = angles[1] * 360
        roll  = angles[2] * 360

        # Mirror correction — webcam flips left/right
        # Without this, turning your head right shows as "Left"
        yaw = -yaw

        return pitch, yaw, roll

    def _get_direction(self, pitch, yaw, roll):
        # Use the largest deviation as the primary direction
        deviations = {
            "yaw":   abs(yaw),
            "pitch": abs(pitch),
            "roll":  abs(roll),
        }
        dominant = max(deviations, key=lambda x: deviations[x])

        if dominant == "yaw" and abs(yaw) > YAW_THRESHOLD:
            return "Right" if yaw > 0 else "Left"
        if dominant == "pitch" and abs(pitch) > PITCH_THRESHOLD:
            return "Down" if pitch > 0 else "Up"
        if dominant == "roll" and abs(roll) > ROLL_THRESHOLD:
            return "Tilted"
        return "Forward"

    def _compute_stability(self):
        if not self.pitch_history:
            return 100
        avg_pitch = np.mean(self.pitch_history)
        avg_yaw   = np.mean(self.yaw_history)
        avg_roll  = np.mean(self.roll_history)
        score     = 100 - (
            min(avg_pitch / PITCH_THRESHOLD, 1.0) * 30 +
            min(avg_yaw   / YAW_THRESHOLD,   1.0) * 45 +
            min(avg_roll  / ROLL_THRESHOLD,  1.0) * 25
        )
        return max(0, int(score))

    def _empty_result(self):
        return {
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "direction": "Unknown", "stability_score": 0, "is_stable": False,
        }


def draw_head_pose_overlay(frame, result):
    direction = result["direction"]
    stability = result["stability_score"]
    dir_color = (0, 255, 0) if direction == "Forward" else (0, 100, 255)

    cv2.putText(frame, f"Head: {direction}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, dir_color, 2)
    cv2.putText(frame,
                f"Pitch:{result['pitch']:.1f}  Yaw:{result['yaw']:.1f}  Roll:{result['roll']:.1f}",
                (10, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, f"Stability: {stability}/100", (10, 192),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if stability >= 60 else (0, 165, 255), 1)
    return frame


def test_on_image(image_path):
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
    estimator = HeadPoseEstimator()
    result    = estimator.detect(lm_result["landmarks"], frame.shape)
    print("\n" + "="*45)
    print("  MODULE 4 - IMAGE TEST RESULT")
    print("="*45)
    print(f"  Direction      : {result['direction']}")
    print(f"  Pitch          : {result['pitch']} deg")
    print(f"  Yaw            : {result['yaw']} deg")
    print(f"  Roll           : {result['roll']} deg")
    print(f"  Stability      : {result['stability_score']}/100")
    out = lm_result["annotated_frame"].copy()
    out = draw_head_pose_overlay(out, result)
    cv2.imshow("Module 4 - Head Pose (any key to close)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    extractor.release()


def test_webcam():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from face_landmarks import FaceLandmarkExtractor
    extractor = FaceLandmarkExtractor()
    estimator = HeadPoseEstimator()
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
            result = estimator.detect(lm_result["landmarks"], frame.shape)
            disp   = draw_head_pose_overlay(disp, result)
        cv2.imshow("Module 4 - Head Pose (Q to quit)", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
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
    print("  MODULE 4 - Head Pose Estimation")
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