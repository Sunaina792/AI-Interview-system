
# MODULE 1: Face Detection & Facial Landmark Extraction
# AI Interview Confidence & Behavior Analysis System


import cv2
import numpy as np
import os
import sys

# ── Safe MediaPipe import (handles all versions) ──────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    NEW_API = True
    print("[INFO] Using NEW MediaPipe API (>= 0.10.x)")
except (ImportError, AttributeError):
    import mediapipe as mp
    NEW_API = False
    print("[INFO] Using LEGACY MediaPipe API (0.9.x)")



# CONFIGURATION

FRAME_WIDTH            = 640
FRAME_HEIGHT           = 480
PROCESS_EVERY_N_FRAMES = 3
EAR_BLINK_THRESHOLD    = 0.20

LANDMARK_INDICES = {
    "left_eye":      [33, 160, 158, 133, 153, 144],
    "right_eye":     [362, 385, 387, 263, 373, 380],
    "left_iris":     [468],
    "right_iris":    [473],
    "nose_tip":      [1],
    "mouth":         [13, 14, 78, 308],
    "left_eyebrow":  [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    "chin":          [152],
    "forehead":      [10],
}

REGION_COLORS = {
    "left_eye":      (0, 255, 0),
    "right_eye":     (0, 255, 0),
    "left_iris":     (255, 100, 0),
    "right_iris":    (255, 100, 0),
    "nose_tip":      (0, 165, 255),
    "mouth":         (0, 0, 255),
    "left_eyebrow":  (255, 255, 0),
    "right_eyebrow": (255, 255, 0),
    "chin":          (255, 0, 255),
    "forehead":      (255, 255, 255),
}



# EXTRACTOR CLASS

class FaceLandmarkExtractor:
    def __init__(self):
        self.frame_count = 0
        self.last_result = None
        if NEW_API:
            self._init_new_api()
        else:
            self._init_legacy_api()

    def _init_new_api(self):
        base_options = mp_python.BaseOptions(model_asset_path=self._get_model_path())
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def _get_model_path(self):
        import urllib.request
        model_path = "face_landmarker.task"
        if not os.path.exists(model_path):
            print("[INFO] Downloading face landmarker model (~6 MB)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] Model downloaded!")
        return model_path

    def _init_legacy_api(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh      = mp.solutions.face_mesh

    def extract(self, frame):
        """For video/webcam — includes frame skipping for performance."""
        self.frame_count += 1
        if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
            return self.last_result if self.last_result else self._empty_result(frame)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        result = self._run_detection(frame)
        self.last_result = result
        return result

    def extract_image(self, frame):
        """For static images — no frame skipping, processes everything."""
        return self._run_detection(frame)

    def _run_detection(self, frame):
        return self._extract_new_api(frame) if NEW_API else self._extract_legacy_api(frame)

    def _extract_legacy_api(self, frame):
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.face_mesh.process(rgb)
        annotated = frame.copy()
        if not results.multi_face_landmarks:
            return self._empty_result(frame)
        face_lms = results.multi_face_landmarks[0]
        self.mp_drawing.draw_landmarks(
            image=annotated, landmark_list=face_lms,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        self.mp_drawing.draw_landmarks(
            image=annotated, landmark_list=face_lms,
            connections=self.mp_face_mesh.FACEMESH_EYES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        key_points = self._to_pixels(face_lms.landmark, frame)
        return {"face_detected": True, "landmarks": face_lms, "key_points": key_points,
                "annotated_frame": annotated, "ear": self._compute_ear(key_points)}

    def _extract_new_api(self, frame):
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self.detector.detect(mp_image)
        annotated = frame.copy()
        if not detection.face_landmarks:
            return self._empty_result(frame)
        landmarks  = detection.face_landmarks[0]
        key_points = self._to_pixels(landmarks, frame)
        return {"face_detected": True, "landmarks": landmarks, "key_points": key_points,
                "annotated_frame": annotated, "ear": self._compute_ear(key_points)}

    def _to_pixels(self, landmark_list, frame):
        h, w = frame.shape[:2]
        key_points = {}
        for region, indices in LANDMARK_INDICES.items():
            pts = []
            for idx in indices:
                if idx < len(landmark_list):
                    lm = landmark_list[idx]
                    pts.append((int(lm.x * w), int(lm.y * h)))
            key_points[region] = pts
        return key_points

    def _compute_ear(self, key_points):
        def ear(pts):
            if len(pts) < 6: return 0.0
            A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
            B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
            return round((A + B) / (2.0 * C), 4) if C != 0 else 0.0
        l = ear(key_points.get("left_eye",  []))
        r = ear(key_points.get("right_eye", []))
        return {"left": l, "right": r, "avg": round((l + r) / 2, 4) if (l and r) else 0.0}

    def _empty_result(self, frame):
        return {"face_detected": False, "landmarks": None, "key_points": {},
                "annotated_frame": frame, "ear": {"left": 0.0, "right": 0.0, "avg": 0.0}}

    def release(self):
        if not NEW_API and hasattr(self, "face_mesh"):
            self.face_mesh.close()



# SHARED DRAWING HELPERS

def draw_key_points(frame, key_points):
    for region, pts in key_points.items():
        color = REGION_COLORS.get(region, (200, 200, 200))
        for pt in pts:
            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, 6, (0, 0, 0), 1)
    return frame

def draw_legend(frame):
    items = [("Eyes",(0,255,0)),("Iris",(255,100,0)),("Nose",(0,165,255)),
             ("Mouth",(0,0,255)),("Eyebrows",(255,255,0)),("Chin/Head",(255,0,255))]
    lx = 10
    ly = frame.shape[0] - (len(items) * 20 + 25)
    cv2.putText(frame, "Legend:", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    for i, (label, color) in enumerate(items):
        y = ly + 18 + i * 20
        cv2.circle(frame, (lx+6, y-5), 5, color, -1)
        cv2.putText(frame, label, (lx+18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
    return frame

def draw_overlay(frame, ear, face_detected):
    color = (0,255,0) if face_detected else (0,0,255)
    text  = "FACE DETECTED" if face_detected else "NO FACE FOUND"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if face_detected:
        blink = "BLINK" if ear["avg"] < EAR_BLINK_THRESHOLD else "Eyes Open"
        cv2.putText(frame, f"EAR: {ear['avg']}  [{blink}]", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, "478 landmarks detected", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    return frame



# MODE 1: IMAGE TEST

def run_image_test(image_path: str, save_output: bool = True):
    """
    Test on a static image file. Saves annotated result next to original.

    Usage:
        python face_landmarks.py --image "D:/photos/myface.jpg"

    Analogy: Like doing a fire drill with a fake alarm before
    the real thing — safe, repeatable, zero risk.
    """
    print("\n" + "="*55)
    print("  IMAGE TEST MODE")
    print("="*55)
    print(f"  File : {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"\n  [ERROR] Cannot load image. Check the path.")
        print(f"  Example: D:/photos/face.jpg")
        return

    print(f"  Size : {frame.shape[1]} x {frame.shape[0]} px\n")

    extractor = FaceLandmarkExtractor()
    result    = extractor.extract_image(frame)

    print("-"*55)

    if not result["face_detected"]:
        print("  [RESULT] NO FACE DETECTED")
        print("\n  Possible fixes:")
        print("    Use a clear, front-facing, well-lit portrait photo")
        print("    Avoid heavy shadows, masks, or extreme head tilt")
        cv2.imshow("Result — No Face Detected (any key to close)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        extractor.release()
        return

    print("  [RESULT] FACE DETECTED\n")

    kp  = result["key_points"]
    ear = result["ear"]

    # Key points table
    print(f"  {'Region':<18} {'Points':>6}   Sample Coord")
    print("  " + "-"*42)
    for region, pts in kp.items():
        sample = str(pts[0]) if pts else "N/A"
        print(f"  {region:<18} {len(pts):>6}   {sample}")

    # EAR report
    print(f"\n  EYE ASPECT RATIO (EAR):")
    print(f"  Left  : {ear['left']:<8}  {'BLINK' if ear['left']  < EAR_BLINK_THRESHOLD else 'Open'}")
    print(f"  Right : {ear['right']:<8}  {'BLINK' if ear['right'] < EAR_BLINK_THRESHOLD else 'Open'}")
    print(f"  Avg   : {ear['avg']:<8}  {'BLINK' if ear['avg']   < EAR_BLINK_THRESHOLD else 'Open'}")

    # Build annotated image
    out = result["annotated_frame"].copy()
    out = draw_key_points(out, kp)
    out = draw_overlay(out, ear, True)
    out = draw_legend(out)

    # Save
    if save_output:
        base, ext = os.path.splitext(image_path)
        out_path  = f"{base}_landmarks{ext}"
        cv2.imwrite(out_path, out)
        print(f"\n  Saved: {out_path}")

    print("\n  Press any key on the window to close.")
    cv2.imshow("Module 1 - Image Test (any key to close)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    extractor.release()
    print("  Done!\n")



# MODE 2: LIVE WEBCAM

def run_webcam_demo():
    """
    Live webcam landmark detection. Press Q to quit.

    Like a magic mirror that draws a precise dot-map
    on your face in real time as you move.
    """
    extractor = FaceLandmarkExtractor()
    cap       = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Ensure it is connected and not in use.")
        return

    print("[INFO] Webcam started. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        result = extractor.extract(frame)
        disp   = result["annotated_frame"].copy()

        if result["face_detected"]:
            disp = draw_key_points(disp, result["key_points"])
            disp = draw_legend(disp)

        disp = draw_overlay(disp, result["ear"], result["face_detected"])
        cv2.imshow("Module 1 - Webcam (Q to quit)", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    extractor.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")



# ENTRY POINT1

if __name__ == "__main__":
    # Command-line: python face_landmarks.py --image "D:\interview_analyzer\test.jpg"
    # Command-line: python face_landmarks.py --webcam
    if len(sys.argv) >= 3 and sys.argv[1] == "--image":
        run_image_test(sys.argv[2])
        sys.exit(0)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--webcam":
        run_webcam_demo()
        sys.exit(0)

    # Interactive menu
    print("\n" + "="*50)
    print("  MODULE 1 - Face Detection & Landmarks")
    print("="*50)
    print("  [1]  Test on an IMAGE file")
    print("  [2]  Live WEBCAM demo")
    print("="*50)
    choice = input("  Enter choice (1 or 2): ").strip()

    if choice == "1":
        path = input("  Enter image path (e.g. D:/photos/face.jpg): ").strip().strip('"')
        run_image_test(path)
    elif choice == "2":
        run_webcam_demo()
    else:
        print("  Invalid. Run again and enter 1 or 2.")