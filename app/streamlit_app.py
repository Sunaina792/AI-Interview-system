# ============================================================
# app/streamlit_app.py — AI Interview Confidence Analyzer
# Uses streamlit-webrtc for real browser webcam access
# Run: streamlit run app/streamlit_app.py
# Install: pip install streamlit-webrtc aiortc
# ============================================================

import streamlit as st
import cv2
import numpy as np
import sys
import os
import time
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ensure project root (one level above `app`) is on sys.path so the
# `modules` package can be imported regardless of the current working
# directory.  Previously the code added the `modules` folder itself which
# caused Python to look for `modules/modules` and resulted in
# ModuleNotFoundError.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from modules.face_landmarks        import FaceLandmarkExtractor
from modules.expression_detection  import ExpressionDetector
from modules.eye_contact           import EyeContactDetector
from modules.head_pose             import HeadPoseEstimator

# ============================================================
# WEBRTC CONFIG — needed for webcam in browser
# ============================================================
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ============================================================
# WEIGHTS
# ============================================================
WEIGHTS = {
    "eye_contact":    30,
    "expression":     25,
    "head_stability": 25,
    "nervousness":    20,
}


def compute_confidence(expr_r, eye_r, head_r):
    eye_s  = eye_r.get("score",             0)
    expr_s = expr_r.get("expression_score", 0)
    head_s = head_r.get("stability_score",  0)
    nerv_s = expr_r.get("nervousness_score", 0)
    nerv_c = max(0, 100 - nerv_s)
    return min(100, max(0, int(
        eye_s  * WEIGHTS["eye_contact"]    / 100 +
        expr_s * WEIGHTS["expression"]     / 100 +
        head_s * WEIGHTS["head_stability"] / 100 +
        nerv_c * WEIGHTS["nervousness"]    / 100
    )))


def confidence_label(score):
    if score >= 80: return "High",     (0, 204, 68)
    if score >= 60: return "Moderate", (0, 170, 255)
    if score >= 40: return "Low",      (0, 102, 255)
    return "Very Low", (0, 34, 255)


def hex_color(score):
    if score >= 80: return "#00cc44"
    if score >= 60: return "#ffaa00"
    if score >= 40: return "#ff6600"
    return "#ff2200"


# ============================================================
# VIDEO PROCESSOR — runs on every webcam frame
# ============================================================
class InterviewProcessor(VideoProcessorBase):
    def __init__(self):
        self.landmark_extractor = FaceLandmarkExtractor()
        self.expr_detector      = ExpressionDetector(fps=20)
        self.eye_detector       = EyeContactDetector()
        self.head_estimator     = HeadPoseEstimator()

        # Shared state (read from main thread)
        self.confidence    = 0
        self.expr_result   = {}
        self.eye_result    = {}
        self.head_result   = {}
        self.face_detected = False
        self.score_history = deque(maxlen=200)
        self.expr_counts   = {}
        self.frame_count   = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        lm_result = self.landmark_extractor.extract(img)
        out       = lm_result["annotated_frame"].copy()

        if lm_result["face_detected"]:
            kp        = lm_result["key_points"]
            landmarks = lm_result["landmarks"]

            self.expr_result = self.expr_detector.detect(kp, img.shape)
            self.eye_result  = self.eye_detector.detect(kp, img.shape)
            self.head_result = self.head_estimator.detect(landmarks, img.shape)

            self.confidence    = compute_confidence(self.expr_result, self.eye_result, self.head_result)
            self.face_detected = True
            self.score_history.append(self.confidence)

            expr = self.expr_result.get("expression", "N/A")
            self.expr_counts[expr] = self.expr_counts.get(expr, 0) + 1

            # Draw on frame
            label, bgr = confidence_label(self.confidence)
            cv2.putText(out, f"Score: {self.confidence}/100  [{label}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
            cv2.putText(out, f"Expr: {self.expr_result.get('expression','N/A')}  "
                             f"Nerv: {self.expr_result.get('nervousness_score',0)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(out, f"Gaze: {self.eye_result.get('gaze_direction','N/A')}  "
                             f"Contact: {self.eye_result.get('eye_contact_pct',0)}%",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(out, f"Head: {self.head_result.get('direction','N/A')}  "
                             f"Stability: {self.head_result.get('stability_score',0)}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        else:
            self.face_detected = False
            cv2.putText(out, "No face detected — position yourself in frame",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)

        return av.VideoFrame.from_ndarray(out, format="bgr24")


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #4a9eff;
    }
    .section-title {
        font-size: 12px;
        color: #4a9eff;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .tip-box {
        background: #1a2a1a;
        border-left: 3px solid #00cc44;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 13px;
        color: #ccffcc;
        margin-bottom: 6px;
    }
    .warn-box {
        background: #2a1a1a;
        border-left: 3px solid #ff6600;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 13px;
        color: #ffccaa;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    st.markdown("### Score Weights")
    w_eye  = st.slider("Eye Contact",    10, 50, 30)
    w_expr = st.slider("Expression",     10, 40, 25)
    w_head = st.slider("Head Stability", 10, 40, 25)
    w_nerv = st.slider("Nervousness",    10, 40, 20)
    total  = w_eye + w_expr + w_head + w_nerv
    if total != 100:
        st.warning(f"Weights sum to {total} (should be 100)")
    st.markdown("---")
    st.markdown("""
    ### How to use
    1. Click **START** on the video widget
    2. Allow camera access in browser
    3. Watch your live confidence score
    4. Press **STOP** when done
    5. See your session summary below
    """)
    st.markdown("---")
    st.markdown("""
    ### Scoring
    - 🟢 80–100 : High
    - 🟡 60–79  : Moderate
    - 🟠 40–59  : Low
    - 🔴 0–39   : Very Low
    """)

# ============================================================
# HEADER
# ============================================================
st.markdown("# 🎯 AI Interview Confidence Analyzer")
st.markdown("Real-time behavioral analysis — eye contact, expression, head pose")
st.markdown("---")

# ============================================================
# LAYOUT
# ============================================================
cam_col, dash_col = st.columns([3, 2])

with cam_col:
    st.markdown('<div class="section-title">📷 Live Feed — Click START below</div>',
                unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="interview-analyzer",
        video_processor_factory=InterviewProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with dash_col:
    st.markdown('<div class="section-title">📊 Live Metrics</div>', unsafe_allow_html=True)

    score_ph   = st.empty()
    metrics_ph = st.empty()
    tips_ph    = st.empty()

st.markdown("---")
st.markdown('<div class="section-title">📈 Confidence Trend</div>', unsafe_allow_html=True)
chart_ph = st.empty()

# ============================================================
# LIVE DASHBOARD — updates while webcam is running
# ============================================================
if ctx.video_processor:
    proc = ctx.video_processor

    while ctx.state.playing:
        confidence = proc.confidence
        expr_r     = proc.expr_result
        eye_r      = proc.eye_result
        head_r     = proc.head_result
        color      = hex_color(confidence)
        label, _   = confidence_label(confidence)

        # Score card
        score_ph.markdown(f"""
        <div class="metric-card" style="border-left-color:{color}">
            <div class="section-title">Confidence Score</div>
            <div style="font-size:32px;font-weight:bold;color:{color}">{confidence}/100</div>
            <div style="font-size:13px;color:#aaa">Level: {label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Module metrics
        expr_val  = expr_r.get("expression",       "N/A")
        nerv_val  = expr_r.get("nervousness_score", 0)
        blink_r   = expr_r.get("blink_rate",        0)
        gaze_val  = eye_r.get("gaze_direction",    "N/A")
        eye_pct   = eye_r.get("eye_contact_pct",    0)
        head_dir  = head_r.get("direction",        "N/A")
        head_stab = head_r.get("stability_score",   0)

        gaze_color = "#00cc44" if gaze_val == "Center"  else "#ff6600"
        head_color = "#00cc44" if head_dir == "Forward" else "#ff6600"

        metrics_ph.markdown(f"""
        <div class="metric-card">
            <div class="section-title">😐 Expression</div>
            <div style="color:#fff;font-size:18px;font-weight:bold">{expr_val}</div>
            <div style="color:#aaa;font-size:13px">Nervousness: {nerv_val}/100 &nbsp;|&nbsp; Blink: {blink_r}/min</div>
        </div>
        <div class="metric-card">
            <div class="section-title">👁️ Eye Contact</div>
            <div style="color:{gaze_color};font-size:18px;font-weight:bold">{gaze_val}</div>
            <div style="color:#aaa;font-size:13px">Contact: {eye_pct}%</div>
        </div>
        <div class="metric-card">
            <div class="section-title">🧠 Head Pose</div>
            <div style="color:{head_color};font-size:18px;font-weight:bold">{head_dir}</div>
            <div style="color:#aaa;font-size:13px">Stability: {head_stab}/100</div>
        </div>
        """, unsafe_allow_html=True)

        # Tips
        tips_html = '<div class="section-title">💡 Live Tips</div>'
        if not proc.face_detected:
            tips_html += '<div class="warn-box">⚠️ No face detected — check your position</div>'
        if gaze_val not in ("Center", "N/A"):
            tips_html += '<div class="warn-box">👀 Look directly at the camera</div>'
        if nerv_val > 40:
            tips_html += '<div class="warn-box">😮‍💨 Take a deep breath, slow down</div>'
        if head_dir not in ("Forward", "N/A", "Unknown"):
            tips_html += '<div class="warn-box">🙆 Keep your head still and facing forward</div>'
        if confidence >= 70 and gaze_val == "Center" and head_dir == "Forward":
            tips_html += '<div class="tip-box">✅ Great job! Keep it up</div>'
        tips_ph.markdown(tips_html, unsafe_allow_html=True)

        # Chart
        scores = list(proc.score_history)
        if len(scores) > 2:
            import pandas as pd
            chart_ph.line_chart(
                pd.DataFrame({"Confidence Score": scores}),
                height=160
            )

        time.sleep(0.1)

# ============================================================
# SESSION SUMMARY
# ============================================================
if ctx.video_processor and not ctx.state.playing:
    proc   = ctx.video_processor
    scores = list(proc.score_history)

    if scores:
        st.markdown("---")
        st.markdown("## 📋 Session Summary")

        avg   = int(np.mean(scores))
        peak  = int(np.max(scores))
        low   = int(np.min(scores))
        label, _ = confidence_label(avg)
        color    = hex_color(avg)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Score",  f"{avg}/100",  label)
        c2.metric("Peak Score", f"{peak}/100")
        c3.metric("Lowest",     f"{low}/100")
        c4.metric("Frames",     proc.frame_count)

        st.markdown("---")
        col_chart, col_expr = st.columns([3, 2])

        with col_chart:
            st.markdown("### Confidence Over Time")
            import pandas as pd
            st.line_chart(pd.DataFrame({"Score": scores}), height=250)

        with col_expr:
            st.markdown("### Expression Breakdown")
            total_expr = sum(proc.expr_counts.values()) or 1
            for expr, count in sorted(proc.expr_counts.items(), key=lambda x: -x[1]):
                pct = int(count / total_expr * 100)
                st.markdown(f"**{expr}** — {pct}%")
                st.progress(pct / 100)

        st.markdown("---")
        st.markdown("### 💡 Improvement Tips")
        t1, t2 = st.columns(2)
        with t1:
            if avg < 40:
                st.error("Practice eye contact — look at the camera, not the screen")
                st.error("Do mock interviews to reduce visible nervousness")
            elif avg < 70:
                st.warning("Focus on consistent eye contact throughout")
                st.warning("Relax your face — a natural expression scores better")
            else:
                st.success("Strong performance! Keep practicing for consistency")
        with t2:
            dominant = max(proc.expr_counts, key=lambda x: proc.expr_counts[x]) if proc.expr_counts else "N/A"
            if dominant in ("Stressed", "Nervous"):
                st.warning("You appeared nervous — try slow breathing before answering")
                st.info("Tip: Pause 2 seconds before every answer to compose yourself")
            elif dominant == "Happy":
                st.success("Great expression — you looked confident and approachable")
            else:
                st.info("Try smiling naturally — it significantly boosts confidence perception")