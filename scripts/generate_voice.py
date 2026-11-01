import os
import torch
from f5_tts.api import F5TTS

def main():
    # voice_text.txt ফাইল থেকে টার্গেট টেক্সট পড়া
    text_file_path = "voice_text.txt"
    
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as f:
            target_text = f.read().strip()
    else:
        target_text = "ডিফল্ট বাংলা ভয়েস জেনারেট হচ্ছে।"
    
    # রেফারেন্স অডিও এবং আউটপুট পাথ
    ref_audio_path = "ref_audio/sample.wav"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_voice.wav")
    
    print(f"[*] টেক্সট ফাইল থেকে পড়া হয়েছে: {target_text}")
    print(f"[*] রেফারেন্স অডিও: {ref_audio_path}")
    
    device = "cpu"
    print(f"[*] ব্যবহৃত ডিভাইস: {device}")

    try:
        print("[*] মডেল লোড হচ্ছে...")
        f5tts = F5TTS(device=device)

        # রেফারেন্স স্ক্রিপ্ট (আপনার রেকর্ড করা অডিওতে যা বলেছিলেন)
        ref_script = """গতকাল রাতে দারুণ একটা থ্রিলার মুভি দেখলাম! এর ব্যাকগ্রাউন্ড মিউজিক এবং শেষ ১৫ মিনিটের টুইস্ট—এককথায় দুর্দান্ত ছিল। যারা সাসপেন্স পছন্দ করেন, তাদের এই সিনেমাটি অবশ্যই দেখা উচিত।"""

        print("[*] ভয়েস জেনারেশন শুরু হয়েছে...")
        f5tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_script,
            gen_text=target_text,
            file_wave_output=output_path,
            nfe_step=32,
            sway_sampling_coef=-1.0
        )
        
        print(f"[✔] সফলভাবে অডিও জেনারেট হয়েছে: {output_path}")

    except Exception as e:
        print(f"[❌] ত্রুটি দেখা দিয়েছে: {e}")
        raise e

if __name__ == "__main__":
    main()
