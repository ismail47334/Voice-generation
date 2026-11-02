import os
import sys
import soundfile as sf

REF_AUDIO = "ref_audio/sample.wav"
REF_TEXT_FILE = "ref_text.txt"
VOICE_TEXT_FILE = "voice_text.txt"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/output_voice_f5.wav"

DEFAULT_REF_TEXT = """গতকাল রাতে দারুণ একটা থ্রিলার মুভি দেখলাম! এর ব্যাকগ্রাউন্ড মিউজিক এবং শেষ ১৫ মিনিটের টুইস্ট—এককথায় দুর্দান্ত ছিল।"""

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_text(path, default=""):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
            if txt:
                print(f"✅ {path}: {len(txt)} chars")
                return txt
    print(f"⚠️ {path} missing, using default")
    return default

def main():
    ref_text = read_text(REF_TEXT_FILE, DEFAULT_REF_TEXT)
    gen_text = read_text(VOICE_TEXT_FILE, "")
    
    if not gen_text:
        print(f"❌ {VOICE_TEXT_FILE} empty!")
        sys.exit(1)
    
    print(f"\nRef Audio exists: {os.path.exists(REF_AUDIO)}")
    print(f"Gen text preview: {gen_text[:100]}...")

    # Import here after deps installed
    import torch
    print(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    
    try:
        from f5_tts.api import F5TTS
        print("Loading F5TTS model F5-TTS_v1_Base...")
        f5 = F5TTS(model="F5-TTS_v1_Base")
        
        print("Starting inference...")
        wav, sr, spec = f5.infer(
            ref_file=REF_AUDIO,
            ref_text=ref_text,
            gen_text=gen_text,
        )
        
        sf.write(OUTPUT_FILE, wav, sr)
        print(f"\n✅ SUCCESS! File: {OUTPUT_FILE}")
        print(f"Size: {os.path.getsize(OUTPUT_FILE)/1024/1024:.2f} MB")
        print(f"Sample rate: {sr}, Duration: {len(wav)/sr:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
