"""
modules/expression_detection.py
ML-based expression detection using trained FER2013 CNN.
Falls back to rule-based if model not found.
"""
import cv2
import numpy as np
import os
import json
from collections import deque

# ── Model path (relative to project root) ────────────────────────────────────
_MODEL_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_MODEL_PATH  = os.path.join(_MODEL_DIR, "best_expression_model.keras")
_CMAP_PATH   = os.path.join(_MODEL_DIR, "expression_class_map.json")

# Nervousness weights per expression (higher = more nervous)
_NERVOUSNESS_WEIGHT = {
    "Angry":    80,
    "Disgust":  70,
    "Fear":     90,
    "Happy":    10,
    "Neutral":  20,
    "Sad":      60,
    "Surprise": 40,
}

# Positive expression score per class (higher = more confident-looking)
_EXPR_SCORE = {
    "Angry":    30,
    "Disgust":  20,
    "Fear":     10,
    "Happy":    95,
    "Neutral":  70,
    "Sad":      25,
    "Surprise": 55,
}


class ExpressionDetector:
    def __init__(self, fps: int = 20):
        self._fps        = fps
        self._model      = None
        self._class_map  = None   # {0: "Angry", 1: "Disgust", ...}
        self._ml_ready   = False

        # Blink tracking (kept for blink_rate metric)
        self._blink_buffer     = deque(maxlen=fps * 10)
        self._blink_count      = 0
        self._blink_frames     = 0
        self._EAR_THRESHOLD    = 0.22
        self._CONSEC_FRAMES    = 2

        # Smoothing buffer for expression predictions
        self._expr_buffer = deque(maxlen=8)

        self._load_model()

    def _load_model(self):
        if not os.path.exists(_MODEL_PATH):
            print(f"[ExpressionDetector] Model not found at {_MODEL_PATH}. Using rule-based fallback.")
            return
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(_MODEL_PATH)
            if os.path.exists(_CMAP_PATH):
                with open(_CMAP_PATH) as f:
                    raw = json.load(f)
                self._class_map = {int(k): v for k, v in raw.items()}
            else:
                # default FER2013 order
                self._class_map = {
                    0: "Angry", 1: "Disgust", 2: "Fear",
                    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise",
                }
            self._ml_ready = True
            print("[ExpressionDetector] ML model loaded successfully.")
        except Exception as e:
            print(f"[ExpressionDetector] Model load failed: {e}. Using rule-based fallback.")

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, key_points: dict, frame_shape: tuple, frame_bgr=None) -> dict:
        """
        Returns:
          expression       : str
          expression_score : int 0-100
          nervousness_score: int 0-100
          blink_rate       : int (blinks/min)
          ml_confidence    : float (model softmax confidence, 0 if rule-based)
        """
        blink_rate = self._update_blink(key_points, frame_shape)

        if self._ml_ready and frame_bgr is not None:
            result = self._ml_detect(key_points, frame_bgr)
        else:
            result = self._rule_based_detect(key_points, frame_shape)

        result["blink_rate"] = blink_rate
        return result

    # ── ML Detection ──────────────────────────────────────────────────────────
    def _ml_detect(self, key_points: dict, frame_bgr: np.ndarray) -> dict:
        face_roi = self._extract_face_roi(key_points, frame_bgr)
        if face_roi is None:
            return self._fallback_neutral()

        try:
            gray    = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            inp     = resized.astype(np.float32) / 255.0
            inp     = inp.reshape(1, 48, 48, 1)

            probs      = self._model.predict(inp, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            ml_conf    = float(probs[pred_idx])

            # Smoothing — majority vote over last N frames
            self._expr_buffer.append(pred_idx)
            from collections import Counter
            smoothed_idx = Counter(self._expr_buffer).most_common(1)[0][0]
            expression   = self._class_map.get(smoothed_idx, "Neutral")

            expr_score = _EXPR_SCORE.get(expression, 50)
            nerv_score = _NERVOUSNESS_WEIGHT.get(expression, 30)

            return {
                "expression":        expression,
                "expression_score":  expr_score,
                "nervousness_score": nerv_score,
                "ml_confidence":     round(ml_conf, 3),
            }
        except Exception as e:
            print(f"[ExpressionDetector] ML inference error: {e}")
            return self._fallback_neutral()

    def _extract_face_roi(self, key_points: dict, frame_bgr: np.ndarray):
        """Crop face region using nose + eye landmarks."""
        try:
            h, w = frame_bgr.shape[:2]
            nose  = key_points.get("nose_tip")
            l_eye = key_points.get("left_eye")
            r_eye = key_points.get("right_eye")
            if nose is None or l_eye is None or r_eye is None:
                return None

            cx = int(nose[0] * w)
            cy = int(nose[1] * h)
            eye_dist = abs(int(l_eye[0] * w) - int(r_eye[0] * w))
            pad = max(eye_dist, 60)

            x1 = max(0, cx - pad)
            y1 = max(0, cy - pad)
            x2 = min(w, cx + pad)
            y2 = min(h, cy + pad)
            roi = frame_bgr[y1:y2, x1:x2]
            return roi if roi.size > 0 else None
        except Exception:
            return None

    # ── Rule-based Fallback ───────────────────────────────────────────────────
    def _rule_based_detect(self, key_points: dict, frame_shape: tuple) -> dict:
        mouth_open   = self._mouth_open_ratio(key_points, frame_shape)
        eyebrow_raise= self._eyebrow_raise(key_points, frame_shape)

        if mouth_open > 0.05 and eyebrow_raise > 0.03:
            expression = "Surprise"
        elif mouth_open > 0.04:
            expression = "Happy"
        elif eyebrow_raise < -0.01:
            expression = "Angry"
        else:
            expression = "Neutral"

        return {
            "expression":        expression,
            "expression_score":  _EXPR_SCORE.get(expression, 50),
            "nervousness_score": _NERVOUSNESS_WEIGHT.get(expression, 30),
            "ml_confidence":     0.0,
        }

    def _fallback_neutral(self) -> dict:
        return {
            "expression":        "Neutral",
            "expression_score":  70,
            "nervousness_score": 20,
            "ml_confidence":     0.0,
        }

    # ── Blink Rate ────────────────────────────────────────────────────────────
    def _update_blink(self, key_points: dict, frame_shape: tuple) -> int:
        ear = self._eye_aspect_ratio(key_points, frame_shape)
        if ear < self._EAR_THRESHOLD:
            self._blink_frames += 1
        else:
            if self._blink_frames >= self._CONSEC_FRAMES:
                self._blink_count += 1
            self._blink_frames = 0

        self._blink_buffer.append(1)
        window_sec = len(self._blink_buffer) / self._fps
        if window_sec > 0:
            return int(self._blink_count / window_sec * 60)
        return 0

    def _eye_aspect_ratio(self, key_points: dict, frame_shape: tuple) -> float:
        try:
            h, w = frame_shape[:2]
            le = key_points.get("left_eye")
            re = key_points.get("right_eye")
            if le is None or re is None:
                return 0.3
            eye_w = abs(le[0] - re[0]) * w
            eye_h = max(le[1], re[1]) * h * 0.15
            return eye_h / (eye_w + 1e-6)
        except Exception:
            return 0.3

    def _mouth_open_ratio(self, key_points: dict, frame_shape: tuple) -> float:
        try:
            h, w = frame_shape[:2]
            top = key_points.get("upper_lip")
            bot = key_points.get("lower_lip")
            if top is None or bot is None:
                return 0.0
            return abs(top[1] - bot[1])
        except Exception:
            return 0.0

    def _eyebrow_raise(self, key_points: dict, frame_shape: tuple) -> float:
        try:
            nose = key_points.get("nose_tip")
            leb  = key_points.get("left_eyebrow")
            reb  = key_points.get("right_eyebrow")
            if nose is None or leb is None or reb is None:
                return 0.0
            avg_brow = (leb[1] + reb[1]) / 2
            return nose[1] - avg_brow
        except Exception:
            return 0.0