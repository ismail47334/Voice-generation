import os
import torch
from TTS.api import TTS

def main():
    text_file_path = "voice_text.txt"
    
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as f:
            target_text = f.read().strip()
    else:
        target_text = "কোলাব থেকে আইডিয়া নিয়ে গিটহাব অ্যাকশনে XTTS v2 চালানো হচ্ছে।"
    
    ref_audio_path = "ref_audio/sample.wav"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_voice.wav")
    
    # ডিভাইস নির্ধারণ (গিটহাবের জন্য এটি স্বয়ংক্রিয়ভাবে cpu হবে)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] ব্যবহৃত ডিভাইস: {device}")

    print("[*] Coqui XTTS v2 মডেল লোড হচ্ছে...")
    # লাইসেন্স গ্রহণ কনফার্ম করার জন্য পরিবেশ ভেরিয়েবল সেট করা
    os.environ["COQUI_TOS_AGREED"] = "1"
    
    # কোলাবের মতো সরাসরি xtts_v2 মডেল লোড
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print("[*] ভয়েস ক্লোনিং ও অডিও জেনারেশন শুরু হয়েছে...")
    
    # XTTS v2 এর মাধ্যমে ভয়েস ক্লোন করে অডিও সেভ করা
    tts.tts_to_file(
        text=target_text,
        file_path=output_path,
        speaker_wav=ref_audio_path,
        language="en"  # বাংলা ভাষার জন্য 'bn'
    )
    
    print(f"[✔] সফলভাবে অডিও জেনারেট হয়েছে: {output_path}")

if __name__ == "__main__":
    main()
