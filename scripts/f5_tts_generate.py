import os
import sys
import soundfile as sf
from pathlib import Path

# Config
REF_AUDIO = "ref_audio/sample.wav"
REF_TEXT_FILE = "ref_text.txt"
VOICE_TEXT_FILE = "voice_text.txt"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/output_voice_f5.wav"

DEFAULT_REF_TEXT = """গতকাল রাতে দারুণ একটা থ্রিলার মুভি দেখলাম! এর ব্যাকগ্রাউন্ড মিউজিক এবং শেষ ১৫ মিনিটের টুইস্ট—এককথায় দুর্দান্ত ছিল। যারা সাসপেন্স পছন্দ করেন, তাদের এই সিনেমাটি অবশ্যই দেখা উচিত।"""

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_text_file(file_path, default_text=""):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    print(f"✅ Read {file_path}: {len(text)} chars")
                    print(f"Preview: {text[:100]}...")
                    return text
        print(f"⚠️ {file_path} not found, using default")
        return default_text
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return default_text

def generate_f5():
    ref_text = read_text_file(REF_TEXT_FILE, DEFAULT_REF_TEXT)
    gen_text = read_text_file(VOICE_TEXT_FILE, "")
    
    if not gen_text:
        print(f"❌ {VOICE_TEXT_FILE} is empty! Add Bengali text")
        sys.exit(1)
    
    print(f"\n=== F5-TTS Config ===")
    print(f"Ref Audio: {REF_AUDIO}")
    print(f"Ref Text: {len(ref_text)} chars")
    print(f"Gen Text: {len(gen_text)} chars")
    
    if not os.path.exists(REF_AUDIO):
        print(f"❌ {REF_AUDIO} not found")
        print("Files in ref_audio:")
        if os.path.exists("ref_audio"):
            for f in os.listdir("ref_audio"):
                print(f" - {f}")
        sys.exit(1)
    
    print("\nLoading F5-TTS model...")
    
    try:
        from f5_tts.api import F5TTS
        
        f5 = F5TTS.from_pretrained("F5-TTS_v1_Base", vocoder_name="vocos", device="cpu")
        
        print("Generating voice...")
        wav, sr, spec = f5.infer(
            ref_file=REF_AUDIO,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=OUTPUT_FILE,
            cross_fade_duration=0.15,
            nfe_step=32,  # Quality: 32 good, 16 fast
            speed=1.0,
            remove_silence=True,
        )
        
        print(f"\n✅ SUCCESS: {OUTPUT_FILE}")
        
        if os.path.exists(OUTPUT_FILE):
            size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
            print(f"Size: {size_mb:.2f} MB")
            print(f"Sample Rate: {sr} Hz")
            print(f"Duration: {len(wav)/sr:.2f} sec")
            
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    generate_f5()
