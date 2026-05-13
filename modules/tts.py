"""
modules/tts.py — Text-to-Speech (Low-Latency)

Strategy:
  1. pyttsx3  — fully offline, zero network latency, engine warmed-up once.
  2. gTTS     — online fallback if pyttsx3 unavailable.
  3. os/wmplayer — last resort Windows fallback.
"""

import threading
import tempfile
import os

# ── pyttsx3 singleton ──────────────────────────────────────────────────────────
_pyttsx3_engine = None
_pyttsx3_lock = threading.Lock()
_pyttsx3_available = False

def _init_pyttsx3():
    """Initialize pyttsx3 engine once at module load (background thread)."""
    global _pyttsx3_engine, _pyttsx3_available
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Tune: rate 160 wpm feels natural for an interviewer
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)

        # Prefer a clear English voice if available
        voices = engine.getProperty('voices')
        for v in voices:
            if 'english' in v.name.lower() or 'zira' in v.name.lower() or 'david' in v.name.lower():
                engine.setProperty('voice', v.id)
                break

        _pyttsx3_engine = engine
        _pyttsx3_available = True
        print('[TTS] pyttsx3 engine ready (offline, low-latency).')
    except Exception as e:
        print(f'[TTS] pyttsx3 unavailable ({e}), will fall back to gTTS.')

# Warm-up in background at import time so first speak() has no init overhead
threading.Thread(target=_init_pyttsx3, daemon=True).start()

# ── pygame mixer singleton (for gTTS fallback) ─────────────────────────────────
_pygame_ready = False

def _ensure_pygame():
    global _pygame_ready
    if _pygame_ready:
        return True
    try:
        import pygame
        pygame.mixer.init()
        _pygame_ready = True
        return True
    except Exception:
        return False


# ── Public API ─────────────────────────────────────────────────────────────────

def speak(text: str) -> str:
    """
    Convert text to speech and play it synchronously.
    Returns the path to audio file (if gTTS path was used) or '' for pyttsx3.

    Latency profile:
      pyttsx3 : ~50-150 ms to first audio sample (fully offline).
      gTTS    : ~800-2000 ms (network + disk save) — only used as fallback.
    """
    if not text or not text.strip():
        return ''

    # ── Path 1: pyttsx3 (fast offline) ────────────────────────────────────────
    with _pyttsx3_lock:
        if _pyttsx3_available and _pyttsx3_engine is not None:
            try:
                _pyttsx3_engine.say(text)
                _pyttsx3_engine.runAndWait()
                return ''
            except Exception as e:
                print(f'[TTS] pyttsx3 speak failed ({e}), trying gTTS...')

    # ── Path 2: gTTS + pygame ─────────────────────────────────────────────────
    tmp_path = ''
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='en', slow=False)
        tmp_path = tempfile.mktemp(suffix='.mp3')
        tts.save(tmp_path)

        if _ensure_pygame():
            import pygame
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
            return tmp_path
    except Exception as e:
        print(f'[TTS] gTTS/pygame failed ({e}), trying wmplayer...')

    # ── Path 3: Windows wmplayer ───────────────────────────────────────────────
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.system(f'start /wait wmplayer "{tmp_path}"')
            return tmp_path
        except Exception:
            pass

    print(f'[TTS] All playback methods failed. Text: {text[:60]}...')
    return tmp_path


def speak_async(text: str) -> threading.Thread:
    """
    Fire-and-forget version of speak().
    Returns the thread so caller can join() if needed.
    """
    t = threading.Thread(target=speak, args=(text,), daemon=True)
    t.start()
    return t