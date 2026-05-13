import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
import time
import argparse
import sys
from collections import deque

SAMPLE_RATE = 22050
CHUNK_DURATION = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION
SILENCE_THRESHOLD = 0.01
SCORE_WINDOW = 5

score_history = deque(maxlen=SCORE_WINDOW)
audio_queue = queue.Queue()


def extract_features(audio: np.ndarray, sr: float = SAMPLE_RATE) -> dict:
    if len(audio) < 512:
        return None

    features = {}

    # RMS energy - volume/projection
    rms = librosa.feature.rms(y=audio)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # Zero crossing rate - voice steadiness
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features["zcr_mean"] = float(np.mean(zcr))

    # Pitch (F0) - monotone vs varied pitch
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_values) > 0:
        features["pitch_mean"] = float(np.mean(pitch_values))
        features["pitch_std"] = float(np.std(pitch_values))
    else:
        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0

    # Speech rate proxy - number of energy bursts
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    features["speech_rate"] = len(onset_frames) / CHUNK_DURATION

    # Pause detection - ratio of silent frames
    silent_frames = np.sum(rms < SILENCE_THRESHOLD)
    features["pause_ratio"] = float(silent_frames / len(rms))

    return features


def compute_audio_score(features: dict) -> dict:
    if features is None:
        return {"score": 0, "tips": ["No audio detected"], "breakdown": {}}

    score = 100
    tips = []
    breakdown = {}

    # 1. Volume/Energy (25 pts)
    rms = features["rms_mean"]
    if rms < 0.02:
        vol_score = 10
        tips.append("Speak louder — your voice is too soft")
    elif rms < 0.05:
        vol_score = 18
        tips.append("Try projecting your voice more confidently")
    elif rms > 0.3:
        vol_score = 18
        tips.append("Slightly lower your volume for a calmer tone")
    else:
        vol_score = 25
    breakdown["volume"] = vol_score

    # 2. Pitch variation (25 pts) - monotone = low confidence
    pitch_std = features["pitch_std"]
    if pitch_std < 10:
        pitch_score = 10
        tips.append("Avoid monotone — vary your pitch to sound engaging")
    elif pitch_std < 30:
        pitch_score = 18
    else:
        pitch_score = 25
    breakdown["pitch_variation"] = pitch_score

    # 3. Speech rate (25 pts)
    rate = features["speech_rate"]
    if rate < 1.5:
        rate_score = 12
        tips.append("You're speaking too slowly — pick up the pace slightly")
    elif rate > 6:
        rate_score = 12
        tips.append("Slow down — speaking too fast signals nervousness")
    else:
        rate_score = 25
    breakdown["speech_rate"] = rate_score

    # 4. Pause ratio (25 pts)
    pause = features["pause_ratio"]
    if pause > 0.6:
        pause_score = 10
        tips.append("Too many pauses - try to maintain a steady flow")
    elif pause > 0.4:
        pause_score = 18
    else:
        pause_score = 25
    breakdown["pauses"] = pause_score

    score = sum(breakdown.values())
    score_history.append(score)
    smoothed = round(float(np.mean(score_history)), 1)

    if not tips:
        tips.append("Voice confidence is good - keep it up!")

    return {
        "score": smoothed,
        "raw_score": score,
        "breakdown": breakdown,
        "tips": tips,
        "features": {
            "rms": round(rms, 4),
            "pitch_std": round(features["pitch_std"], 2),
            "speech_rate": round(features["speech_rate"], 2),
            "pause_ratio": round(features["pause_ratio"], 2),
        },
    }


def get_label(score: float) -> str:
    if score >= 75:
        return "Confident"
    elif score >= 50:
        return "Moderate"
    else:
        return "Needs Improvement"


def analyze_file(path: str):
    print(f"\nLoading: {path}")
    try:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    total_chunks = len(audio) // CHUNK_SAMPLES
    if total_chunks == 0:
        print("Audio too short (need at least 2 seconds)")
        sys.exit(1)

    print(f"Duration: {len(audio)/sr:.1f}s | Chunks: {total_chunks}\n")
    all_scores = []

    for i in range(total_chunks):
        chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
        features = extract_features(chunk, sr)
        result = compute_audio_score(features)
        all_scores.append(result["score"])

        print(f"[Chunk {i+1}/{total_chunks}]")
        print(f"  Score     : {result['score']} — {get_label(result['score'])}")
        print(f"  Volume    : {result['breakdown'].get('volume', 0)}/25")
        print(f"  Pitch Var : {result['breakdown'].get('pitch_variation', 0)}/25")
        print(f"  Rate      : {result['breakdown'].get('speech_rate', 0)}/25")
        print(f"  Pauses    : {result['breakdown'].get('pauses', 0)}/25")
        print(f"  Tip       : {result['tips'][0]}")
        print()

    final = round(float(np.mean(all_scores)), 1)
    print("=" * 45)
    print(f"FINAL AUDIO CONFIDENCE SCORE: {final}/100")
    print(f"Overall: {get_label(final)}")
    print("=" * 45)


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())


def analyze_mic():
    print("\nMic mode started. Press Ctrl+C to stop.\n")
    buffer = np.array([], dtype=np.float32)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                chunk_data = audio_queue.get()
                buffer = np.append(buffer, chunk_data.flatten())

                if len(buffer) >= CHUNK_SAMPLES:
                    chunk = buffer[:CHUNK_SAMPLES]
                    buffer = buffer[CHUNK_SAMPLES:]

                    features = extract_features(chunk)
                    result = compute_audio_score(features)

                    print(f"\rScore: {result['score']:5.1f} | {get_label(result['score']):<20} | Tip: {result['tips'][0][:50]}", end="", flush=True)

        except KeyboardInterrupt:
            print("\n\nSession ended.")
            if score_history:
                final = round(float(np.mean(score_history)), 1)
                print(f"Session Avg Score: {final}/100 — {get_label(final)}")


def get_latest_result() -> dict:
    """Called by fusion_scoring.py or Streamlit to get current audio score."""
    if not score_history:
        return {"score": 0, "tips": ["No audio data yet"], "breakdown": {}}
    return {"score": round(float(np.mean(score_history)), 1)}


def process_frame_audio(audio_chunk: np.ndarray) -> dict:
    """Called per-frame from main.py for real-time integration."""
    features = extract_features(audio_chunk)
    return compute_audio_score(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Confidence Analyzer")
    parser.add_argument("--mic", action="store_true", help="Live mic analysis")
    parser.add_argument("--file", type=str, help="Path to audio/video file")
    args = parser.parse_args()

    if args.mic:
        analyze_mic()
    elif args.file:
        analyze_file(args.file)
    else:
        print("\nSelect mode:")
        print("1. Live Microphone")
        print("2. Audio File")
        choice = input("Enter choice (1/2): ").strip()

        if choice == "1":
            analyze_mic()
        elif choice == "2":
            path = input("Enter file path: ").strip()
            analyze_file(path)
        else:
            print("Invalid choice")
            sys.exit(1)