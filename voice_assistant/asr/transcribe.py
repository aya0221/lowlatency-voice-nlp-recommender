import whisper
import sys
import argparse
from pathlib import Path

# Enable root-level imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def transcribe_audio(file_path: str) -> str:
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Transcribing...")
    result = model.transcribe(file_path)
    transcription = result["text"]
    print("Transcription:", transcription)
    return transcription

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    transcribe_audio(args.file)
