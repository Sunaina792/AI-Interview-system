"""
modules/stt.py — Speech-to-Text
Groq Whisper API (whisper-large-v3-turbo) — fast cloud transcription.

Improvements:
  - Energy-based silence detection — skips API call on near-silent audio.
  - Interview-context prompt for better accuracy.
  - Hallucination suppression for common Whisper artifacts.
  - Explicit English language hint.
"""

import os
import tempfile
import numpy as np
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment.")

client = Groq(api_key=GROQ_API_KEY)

SAMPLE_RATE    = 16_000
RECORD_SECONDS = 45
MODEL          = "whisper-large-v3-turbo"

# RMS energy threshold — below this is treated as silence (skip API call)
SILENCE_THRESHOLD = 0.002

# Context prompt that primes Whisper for interview-style speech
INTERVIEW_PROMPT = (
    "This is a professional job interview. The candidate is speaking in English "
    "about their technical skills, work experience, and career goals."
)

# Common Whisper hallucinations on near-silent audio — suppress these
HALLUCINATIONS = {
    '', 'thank you.', 'thanks for watching.', 'thank you for watching.',
    'thanks.', 'you', '.', 'bye.', 'bye bye.', 'thank you very much.',
    'please subscribe.', 'subscribe.',
}


def _is_silent(audio_np: np.ndarray) -> bool:
    """Return True if audio RMS energy is below the silence threshold."""
    rms = float(np.sqrt(np.mean(audio_np.astype(np.float32) ** 2)))
    print(f'[STT] Audio RMS energy: {rms:.5f} (threshold: {SILENCE_THRESHOLD})')
    return rms < SILENCE_THRESHOLD


def _record_audio(duration: int = RECORD_SECONDS) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError("sounddevice not installed. Run: pip install sounddevice")

    print(f'[STT] Recording for {duration}s...')
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
    )
    sd.wait()
    print('[STT] Recording complete.')
    return audio.flatten()


def _numpy_to_wav_bytes(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    import struct
    import io

    num_samples     = len(audio_np)
    num_channels    = 1
    bits_per_sample = 16
    byte_rate       = sample_rate * num_channels * bits_per_sample // 8
    block_align     = num_channels * bits_per_sample // 8
    data_size       = num_samples * block_align

    audio_int16 = (audio_np * 32767).astype(np.int16)

    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<H', num_channels))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', byte_rate))
    buf.write(struct.pack('<H', block_align))
    buf.write(struct.pack('<H', bits_per_sample))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio_int16.tobytes())

    return buf.getvalue()


def transcribe(audio_path: str | None = None, duration: int = RECORD_SECONDS) -> str:
    """
    Transcribe speech to text via Groq Whisper API.

    Args:
        audio_path: Path to an existing audio file. If None, records from mic.
        duration:   Seconds to record if audio_path is None.

    Returns:
        Transcribed text string (empty string on silence or failure).
    """
    try:
        if audio_path is not None:
            # ── Silence check on file ──────────────────────────────────────────
            try:
                import soundfile as sf
                audio_np, _ = sf.read(audio_path, dtype='float32')
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if _is_silent(audio_np):
                    print('[STT] Audio is silent — skipping transcription.')
                    return ''
            except Exception:
                pass  # If we can't read it for silence check, let Groq try anyway

            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            filename = os.path.basename(audio_path)

        else:
            # ── Live mic recording ─────────────────────────────────────────────
            audio_np = _record_audio(duration)
            if _is_silent(audio_np):
                print('[STT] Recorded audio is silent — skipping transcription.')
                return ''
            audio_bytes = _numpy_to_wav_bytes(audio_np)
            filename    = 'recording.wav'

        response = client.audio.transcriptions.create(
            model=MODEL,
            file=(filename, audio_bytes),
            response_format='text',
            language='en',              # force English — avoids misdetection lag
            prompt=INTERVIEW_PROMPT,    # context conditioning for accuracy
        )

        text = response.strip() if isinstance(response, str) else (response.text or '').strip()

        # ── Hallucination suppression ──────────────────────────────────────────
        if text.lower() in HALLUCINATIONS:
            print(f'[STT] Suppressed hallucination: "{text}"')
            return ''

        print(f'[STT] Transcribed ({len(text)} chars): "{text[:100]}"')
        return text

    except Exception as e:
        print(f'[STT] Groq transcription error: {e}')
        return ''