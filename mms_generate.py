import os
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer

def main():
    text_file_path = "voice_text.txt"
    
    if os.path.exists(text_file_path):
        with open(text_file_path, "r", encoding="utf-8") as f:
            target_text = f.read().strip()
    else:
        target_text = "ফেসবুকের এমএমএস মডেল দিয়ে বাংলা ভয়েস জেনারেট হচ্ছে।"
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output_voice.wav")
    
    print("[*] Facebook MMS-TTS মডেল এবং টোকেনাইজার লোড হচ্ছে...")
    
    # মেটার অফিসিয়াল বাংলা এমএমএস মডেল
    model_id = "facebook/mms-tts-ben"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    
    print("[*] ভয়েস জেনারেশন শুরু হয়েছে...")
    inputs = tokenizer(target_text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).waveform
        
    # অডিও ফাইল সেভ করা (ಸಾధারণত 16000Hz স্যামপ্লিঙ রেট থাকে)
    sf.write(output_path, output.numpy().T, model.config.sampling_rate)
    
    print(f"[✔] সফলভাবে অডিও জেনারেট হয়েছে: {output_path}")

if __name__ == "__main__":
    main()
