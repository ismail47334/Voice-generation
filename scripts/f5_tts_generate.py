import os
import sys
import soundfile as sf

REF_AUDIO = "ref_audio/sample.wav"
REF_TEXT_FILE = "ref_text.txt"
VOICE_TEXT_FILE = "voice_text.txt"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/output_voice_f5.wav"

DEFAULT_REF_TEXT = "গতকাল রাতে দারুণ একটা থ্রিলার মুভি দেখলাম! এর ব্যাকগ্রাউন্ড মিউজিক এবং শেষ ১৫ মিনিটের টুইস্ট এককথায় দুর্দান্ত ছিল।"

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
        print(f"❌ {VOICE_TEXT_FILE} খালি!")
        sys.exit(1)
    
    print(f"\nRef Audio exists: {os.path.exists(REF_AUDIO)}")
    print(f"Gen text: {gen_text[:120]}...")
    
    import torch
    print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    try:
        from f5_tts.api import F5TTS
        print("\n--- Loading F5TTS model: F5TTS_v1_Base (no hyphen!) ---")
        # FIX: Correct model name is F5TTS_v1_Base, NOT F5-TTS_v1_Base
        f5tts = F5TTS(model="F5TTS_v1_Base", device="cpu")
        
        print("Model loaded! Starting inference (may download ~1.5GB first time)...")
        wav, sr, spec = f5tts.infer(
            ref_file=REF_AUDIO,
            ref_text=ref_text,
            gen_text=gen_text,
        )
        
        sf.write(OUTPUT_FILE, wav, sr)
        print(f"\n✅ SUCCESS! File saved: {OUTPUT_FILE}")
        print(f"Size: {os.path.getsize(OUTPUT_FILE)/1024/1024:.2f} MB")
        print(f"Sample rate: {sr}, Duration: {len(wav)/sr:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        # Try fallback without model arg (uses default)
        print("\n--- Trying fallback with default model ---")
        try:
            from f5_tts.api import F5TTS
            f5tts = F5TTS()
            wav, sr, spec = f5tts.infer(
                ref_file=REF_AUDIO,
                ref_text=ref_text,
                gen_text=gen_text,
            )
            sf.write(OUTPUT_FILE, wav, sr)
            print(f"✅ Fallback SUCCESS: {OUTPUT_FILE}")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
