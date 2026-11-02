import os
import sys
import subprocess
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
    print(f"Gen Text length: {len(gen_text)} chars")
    
    if not os.path.exists(REF_AUDIO):
        print(f"❌ {REF_AUDIO} not found")
        sys.exit(1)

    # Method 1: Try CLI (most stable for 1.1.6)
    print("\nTrying F5-TTS CLI method...")
    try:
        # Escape quotes in texts
        cmd = [
            "f5-tts_infer-cli",
            "--model", "F5-TTS_v1_Base",
            "--ref_audio", REF_AUDIO,
            "--ref_text", ref_text,
            "--gen_text", gen_text,
            "--output_dir", OUTPUT_DIR,
            "--output_file", "output_voice_f5.wav",
            "--remove_silence",
            "--nfe_step", "32",
        ]
        print(f"Running: {' '.join(cmd[:5])} ...")
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=600)
        
        if os.path.exists(OUTPUT_FILE):
            print(f"\n✅ SUCCESS via CLI: {OUTPUT_FILE}")
            size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
            print(f"Size: {size_mb:.2f} MB")
            return
        
        print("CLI didn't produce file, trying API v2...")
    except Exception as e:
        print(f"CLI failed: {e}")
        import traceback
        traceback.print_exc()

    # Method 2: API v2 for 1.1.6
    print("\nTrying F5-TTS API v2...")
    try:
        from f5_tts.api import F5TTS
        
        # In 1.1.6, it's F5TTS() not from_pretrained
        f5 = F5TTS(
            model_type="F5-TTS_v1_Base",
            ckpt_file="",  # will auto download
            vocab_file="",
            ode_method="euler",
            use_ema=True,
            device="cpu"
        )
        
        wav, sr, spec = f5.infer(
            ref_file=REF_AUDIO,
            ref_text=ref_text,
            gen_text=gen_text,
        )
        
        # Save
        sf.write(OUTPUT_FILE, wav, sr)
        print(f"\n✅ SUCCESS via API v2: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"\n❌ API v2 failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Method 3: Legacy low-level API
        print("\nTrying Legacy Low-level API...")
        try:
            from f5_tts.model import DiT
            from f5_tts.infer.utils_infer import (
                load_vocoder,
                load_model,
                preprocess_ref_audio_text,
                infer_process
            )
            
            vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device="cpu")
            model = load_model(
                model_cls=DiT,
                model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                ckpt_path="hf://SWivid/F5-TTS/F5-TTS_v1_Base/model_1200000.safetensors",
                device="cpu"
            )
            
            ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(REF_AUDIO, ref_text)
            
            final_wave, final_sample_rate, combined_spectrogram = infer_process(
                ref_audio_proc,
                ref_text_proc,
                gen_text,
                model,
                vocoder,
                cross_fade_duration=0.15,
                nfe_step=32,
                speed=1.0,
                device="cpu"
            )
            
            sf.write(OUTPUT_FILE, final_wave, final_sample_rate)
            print(f"\n✅ SUCCESS via Legacy API: {OUTPUT_FILE}")
            
        except Exception as e2:
            print(f"\n❌ All methods failed: {e2}")
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    generate_f5()
