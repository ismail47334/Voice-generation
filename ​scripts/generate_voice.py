import os
import torch
from f5_tts.api import F5TTS

def main():
    # এনভায়রনমেন্ট ভেরিয়েবল থেকে টেক্সট সংগ্রহ
    target_text = os.getenv("TARGET_TEXT", "বাংলা ভয়েস জেনারেট হচ্ছে।")
    
    # ইনপুট এবং আউটপুট পাথ
    ref_audio_path = "ref_audio/sample.wav"
    output_dir = "output"
    os.makedirs(output_dir, SDK=True) if hasattr(os, 'SDK') else os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_voice.wav")
    
    print(f"[*] লক্ষ্য টেক্সট: {target_text}")
    print(f"[*] রেফারেন্স অডিও: {ref_audio_path}")
    
    # ডিভাইস নির্ধারণ (GitHub Actions CPU রানারের জন্য)
    device = "cpu"
    print(f"[*] ব্যবহৃত ডিভাইস: {device}")

    try:
        # F5-TTS মডেল ইনিশিয়ালাইজ করা (ডিফল্ট চেকপয়েন্ট ব্যবহার করে)
        print("[*] মডেল লোড হচ্ছে...")
        f5tts = F5TTS(device=device)

        # রেফারেন্স অডিওর সাথে মিল রেখে ভয়েস জেনারেট করা
        # (নোট: ref_text এ আপনার রেফারেন্স অডিওতে ঠিক যা বলা হয়েছে তা দিতে হবে, 
        # অথবা খালি রাখলে মডেল অটো-ডিটেক্ট বা বেসিক জেনারেশন করবে)
        print("[*] ভয়েস জেনারেশন শুরু হয়েছে...")
        
        f5tts.infer(
            ref_file=ref_audio_path,
            ref_text="", # রেফারেন্স অডিওর স্ক্রিপ্ট জানা থাকলে এখানে দিতে পারেন
            gen_text=target_text,
            file_wave_output=output_path,
            nfe_step=32,   # CPU-তে স্পিড অপ্টিমাইজ করতে স্টেপ কমিয়ে রাখা যেতে পারে (ডিফল্ট ৩২ বা ১৬)
            sway_sampling_coef=-1.0
        )
        
        print(f"[✔] সফলভাবে অডিও জেনারেট হয়েছে: {output_path}")

    except Exception as e:
        print(f"[❌] ত্রুটি দেখা দিয়েছে: {e}")
        raise e

if __name__ == "__main__":
    main()
