import subprocess
import whisper
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from voice_assistant.asr.transcribe import transcribe_audio

def record_and_transcribe(path="voice_assistant/data/input.wav", duration=5):
    print("[INFO] Recording audio...")
    record_cmd = [
        "ffmpeg", "-y", "-f", "alsa", "-i", "default", "-t", str(duration), path
    ]
    subprocess.run(record_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[INFO] Running Whisper ASR...")
    return transcribe_audio(path)
