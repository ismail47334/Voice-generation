import os
import sys
import soundfile as sf
from pathlib import Path

# Config - Voice-generation repo structure
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
        print(f"❌ {VOICE_TEXT_FILE} is empty!")
        sys.exit(1)
    
    print(f"\n=== F5-TTS Config ===")
    print(f"Ref Audio: {REF_AUDIO}")
    print(f"Ref Text: {ref_text[:80]}...")
    print(f"Gen Text length: {len(gen_text)} chars")
    
    if not os.path.exists(REF_AUDIO):
        print(f"❌ {REF_AUDIO} not found")
        sys.exit(1)

    # Use F5-TTS API directly - compatible with numpy 2.x
    print("\nTrying F5-TTS API...")
    try:
        # Fix: Import must happen AFTER numpy check
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        from f5_tts.api import F5TTS
        
        # In f5-tts 1.1.6, correct init
        f5 = F5TTS(
            model="F5-TTS_v1_Base",
            device="cpu"
        )
        
        print("Model loaded, starting inference...")
        wav, sr, spec = f5.infer(
            ref_file=REF_AUDIO,
            ref_text=ref_text,
            gen_text=gen_text,
            # Better quality params
            speed=1.0,
            nfe_step=32,
        )
        
        # Save
        sf.write(OUTPUT_FILE, wav, sr)
        print(f"\n✅ SUCCESS! Generated: {OUTPUT_FILE}")
        size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
        print(f"Size: {size_mb:.2f} MB, Sample Rate: {sr}")
        
        # Also save as final output for artifact
        print(f"\nDuration: {len(wav)/sr:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"\n❌ F5-TTS API failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    generate_f5()
