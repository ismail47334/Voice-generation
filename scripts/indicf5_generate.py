import os
import sys
import numpy as np
import soundfile as sf

# Your files
REF_AUDIO = "ref_audio/sample.wav"
REF_TEXT_FILE = "ref_text.txt"
VOICE_TEXT_FILE = "voice_text.txt"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/output_voice_bangla.wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_text(path, default=""):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
            if txt:
                print(f"✅ {path}: {len(txt)} chars")
                return txt
    print(f"⚠️ {path} not found")
    return default

def main():
    print("=== IndicF5 Bengali TTS ===")
    ref_text = read_text(REF_TEXT_FILE)
    gen_text = read_text(VOICE_TEXT_FILE)

    if not gen_text:
        print(f"❌ {VOICE_TEXT_FILE} empty!")
        sys.exit(1)
    if not os.path.exists(REF_AUDIO):
        print(f"❌ {REF_AUDIO} not found!")
        sys.exit(1)

    print(f"\nRef Audio: {REF_AUDIO}")
    print(f"Ref Text: {ref_text[:100]}...")
    print(f"Gen Text: {gen_text[:150]}...")

    try:
        from transformers import AutoModel
        print("\n--- Loading ai4bharat/IndicF5 (Bengali supported, ~1.5GB download first time) ---")
        # Official method from https://github.com/AI4Bharat/IndicF5
        model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

        print("Generating Bengali voice...")
        # IndicF5 usage: audio = model(text, ref_audio_path, ref_text)
        audio = model(
            gen_text,
            ref_audio_path=REF_AUDIO,
            ref_text=ref_text,
        )

        print(f"Raw audio type: {type(audio)}, dtype: {audio.dtype if hasattr(audio, 'dtype') else 'unknown'}")

        # Normalize as per official README
        if hasattr(audio, 'dtype') and audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Ensure numpy array
        audio_np = np.array(audio, dtype=np.float32)
        
        # Save
        sf.write(OUTPUT_FILE, audio_np, samplerate=24000)
        
        print(f"\n✅ SUCCESS! Bengali voice generated: {OUTPUT_FILE}")
        print(f"Size: {os.path.getsize(OUTPUT_FILE)/1024/1024:.2f} MB")
        print(f"Duration: {len(audio_np)/24000:.2f} seconds")

    except Exception as e:
        print(f"\n❌ IndicF5 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
