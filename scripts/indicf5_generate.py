import os
import sys
import numpy as np
import soundfile as sf

REF_AUDIO = "ref_audio/sample.wav"
REF_TEXT_FILE = "ref_text.txt"
VOICE_TEXT_FILE = "voice_text.txt"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/output_voice_bangla.wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_text(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

def main():
    print("=== IndicF5 Bengali TTS - FIXED for Gated Repo ===")
    ref_text = read_text(REF_TEXT_FILE)
    gen_text = read_text(VOICE_TEXT_FILE)

    if not gen_text:
        print("❌ voice_text.txt empty!")
        sys.exit(1)

    # === FIX: Login to HuggingFace if token provided ===
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print("🔑 HF_TOKEN found, logging in...")
        try:
            from huggingface_hub import login
            login(token=hf_token)
            print("✅ HuggingFace login OK")
        except Exception as e:
            print(f"⚠️ Login failed: {e}")
    else:
        print("⚠️ No HF_TOKEN found - trying without login (may fail if gated)")

    try:
        from transformers import AutoModel
        print("\n--- Loading ai4bharat/IndicF5 ---")
        model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

        print("Generating...")
        audio = model(
            gen_text,
            ref_audio_path=REF_AUDIO,
            ref_text=ref_text,
        )

        if hasattr(audio, 'dtype') and audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        audio_np = np.array(audio, dtype=np.float32)
        sf.write(OUTPUT_FILE, audio_np, samplerate=24000)
        
        print(f"\n✅ SUCCESS: {OUTPUT_FILE}")
        print(f"Duration: {len(audio_np)/24000:.2f}s")

    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n--- POSSIBLE FIXES ---")
        print("1. Go to https://huggingface.co/ai4bharat/IndicF5 and click 'Agree and access repository'")
        print("2. Create token at https://huggingface.co/settings/tokens")
        print("3. Add token as HF_TOKEN secret in GitHub: Settings > Secrets and variables > Actions > New repository secret > Name: HF_TOKEN")
        sys.exit(1)

if __name__ == "__main__":
    main()
