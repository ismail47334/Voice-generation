import os, wave
from piper import PiperVoice

INPUT_TXT = "voice_text.txt"
OUTPUT_WAV = "output/output_voice_bangla_sherpa.wav"
MODEL = "vits-piper-bn/bn_BD_nabanita_medium.onnx"

os.makedirs("output", exist_ok=True)

with open(INPUT_TXT, 'r', encoding='utf-8') as f:
    text = f.read().strip()

print(f"Text: {len(text)} chars")
voice = PiperVoice.load(MODEL)
print("✅ Model loaded")

with wave.open(OUTPUT_WAV, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(22050)
    voice.synthesize_wav(text, wav_file)

print(f"✅✅✅ SUCCESS: {OUTPUT_WAV}")
