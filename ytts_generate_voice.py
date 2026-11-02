import os
import subprocess
from TTS.api import TTS

def preprocess_audio(input_path, output_path):
    """FFmpeg দিয়ে রেফারেন্স অডিওর কোয়ালিটি অপ্টিমাইজ করা (Mono ও 22050Hz)"""
    print("[*] রেফারেন্স অডিও অপ্টিমাইজ করা হচ্ছে...")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "22050", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    text_file_path = "voice_text.txt"
    
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as f:
            target_text = f.read().strip()
    else:
        target_text = "বাংলা ভয়েস জেনারেট হচ্ছে।"
    
    raw_ref_audio = "ref_audio/sample.wav"
    clean_ref_audio = "ref_audio/clean_sample.wav"
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_voice.wav")
    
    # অডিও প্রি-প্রসেসিং কল করা
    if os.path.exists(raw_ref_audio):
        preprocess_audio(raw_ref_audio, clean_ref_audio)
    else:
        clean_ref_audio = raw_ref_audio

    print("[*] TTS মডেল লোড হচ্ছে (CPU মোড)...")
    
    # Coqui TTS-এর সঠিক মাল্টিলিঙ্গুয়াল মডেল নেম ফরম্যাট
    model_name = "tts_models/bn/custom/vits"  # অথবা জেনেরিক মাল্টিলিঙ্গুয়াল পাথ
    
    try:
        tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False)
    except Exception:
        # যদি ডিফল্ট মডেল লোড করতে সমস্যা হয়, সরাসরি মাল্টি-স্পিকার মডেল ব্যবহার
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/bark", progress_bar=False, gpu=False)

    print("[*] ভয়েস ক্লোনিং ও অডিও জেনারেশন শুরু হয়েছে...")
    
    tts.tts_to_file(
        text=target_text,
        speaker_wav=clean_ref_audio,
        file_path=output_path
    )
    
    print(f"[✔] সফলভাবে অডিও জেনারেট হয়েছে: {output_path}")

if __name__ == "__main__":
    main()
