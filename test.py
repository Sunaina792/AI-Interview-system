# test.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.stt import transcribe
from modules.tts import speak
from modules.llm import generate_questions, evaluate_answer

# Test LLM
print("Testing LLM...")
qs = generate_questions('Sunaina', 'ML Engineer', '1 year', 'Python, OpenCV')
for i, q in enumerate(qs, 1):
    print(f"{i}. {q}")

# Test TTS
print("\nTesting TTS...")
path = speak("Hello! This is a TTS test.")
print(f"Audio saved at: {path}")

print("\nAll tests passed!")