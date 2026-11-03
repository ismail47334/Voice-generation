import os
import sys
import wave
import numpy as np

# Config - আপনার existing file গুলো ব্যবহার করবে
INPUT_TXT = "voice_text.txt"  # আপনার repo তে এই ফাইলটাই আছে
OUTPUT_WAV = "output/output_voice_bangla_sherpa.wav"
MODEL_DIR = "vits-piper-bn"

os.makedirs("output", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def merge_wav_files(wav_files, output_path):
    """Multiple wav কে একটা বড় wav এ জোড়া দেওয়া"""
    if not wav_files:
        return
    with wave.open(wav_files[0], 'rb') as w:
        params = w.getparams()
    
    all_frames = []
    for f in wav_files:
        with wave.open(f, 'rb') as w:
            all_frames.append(w.readframes(w.getnframes()))
    
    with wave.open(output_path, 'wb') as out:
        out.setparams(params)
        for frame in all_frames:
            out.writeframes(frame)

def main():
    print("=== Sherpa-ONNX Bengali TTS (Lightweight & Fast) ===")
    
    if not os.path.exists(INPUT_TXT):
        print(f"❌ {INPUT_TXT} not found!")
        sys.exit(1)
    
    with open(INPUT_TXT, 'r', encoding='utf-8') as f:
        full_text = f.read().strip()
    
    if not full_text:
        print(f"❌ {INPUT_TXT} empty!")
        sys.exit(1)
    
    print(f"📄 Text: {len(full_text)} chars")
    print(f"📄 Preview: {full_text[:100]}...")
    
    # --- Load Model ---
    print("\n--- Loading Sherpa-ONNX model ---")
    try:
        from sherpa_onnx import OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsVitsModelConfig
    except ImportError:
        print("Installing sherpa-onnx...")
        os.system("pip install sherpa-onnx")
        from sherpa_onnx import OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig, OfflineTtsVitsModelConfig

    # Find model files automatically (different versions have different names)
    onnx_file = None
    tokens_file = None
    data_dir = None
    
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".onnx") and onnx_file is None:
                onnx_file = os.path.join(root, file)
            if file == "tokens.txt" and tokens_file is None:
                tokens_file = os.path.join(root, file)
        for d in dirs:
            if "espeak" in d.lower() and data_dir is None:
                data_dir = os.path.join(root, d)
    
    # Fallback paths
    if not onnx_file:
        # common names
        candidates = [f"{MODEL_DIR}/bn_BD_nabanita_medium.onnx", f"{MODEL_DIR}/en_BN_nabanita_piper_phonetic.onnx", f"{MODEL_DIR}/model.onnx"]
        for c in candidates:
            if os.path.exists(c):
                onnx_file = c
                break
    
    if not tokens_file:
        tokens_file = f"{MODEL_DIR}/tokens.txt"
    if not data_dir:
        data_dir = f"{MODEL_DIR}/espeak-ng-data"
        if not os.path.exists(data_dir):
            data_dir = ""

    print(f"Model: {onnx_file}")
    print(f"Tokens: {tokens_file}")
    print(f"Data dir: {data_dir}")

    if not os.path.exists(onnx_file or ""):
        print(f"❌ ONNX model not found in {MODEL_DIR}")
        print(f"Listing {MODEL_DIR}: {os.listdir(MODEL_DIR)}")
        sys.exit(1)

    vits_config = OfflineTtsVitsModelConfig(
        model=onnx_file,
        lexicon="",
        tokens=tokens_file,
        data_dir=data_dir if data_dir and os.path.exists(data_dir) else "",
        dict_dir="",
        length_scale=1.0
    )
    
    model_config = OfflineTtsModelConfig(
        vits=vits_config,
        num_threads=2,
        debug=False,
        provider="cpu"
    )
    
    tts_config = OfflineTtsConfig(
        model=model_config,
        rule_fsts="",
        max_num_sentences=2
    )
    
    tts = OfflineTts(tts_config)
    print("✅ TTS Engine loaded")

    # --- Generate: Chunk + Merge (Best for long text) ---
    # Sherpa handles long text better if split by sentence
    import re
    sentences = re.split(r'[।\n]+', full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print(f"\n✂️ Split into {len(sentences)} sentences")
    
    temp_dir = "temp_sherpa"
    os.makedirs(temp_dir, exist_ok=True)
    temp_files = []
    
    for i, sent in enumerate(sentences):
        if not sent:
            continue
        temp_path = f"{temp_dir}/part_{i}.wav"
        print(f"[{i+1}/{len(sentences)}] Generating: {sent[:40]}...")
        audio = tts.generate(sent, sid=0, speed=1.0)
        if len(audio.samples) == 0:
            print(f"⚠️ Empty audio for: {sent}")
            continue
        audio.save(temp_path)
        temp_files.append(temp_path)
    
    print(f"\n🔗 Merging {len(temp_files)} parts into single file...")
    
    if not temp_files:
        print("❌ No audio generated!")
        sys.exit(1)
    
    if len(temp_files) == 1:
        import shutil
        shutil.copy(temp_files[0], OUTPUT_WAV)
    else:
        merge_wav_files(temp_files, OUTPUT_WAV)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    if os.path.exists(OUTPUT_WAV):
        size_mb = os.path.getsize(OUTPUT_WAV) / 1024 / 1024
        print(f"\n✅✅✅ SUCCESS: {OUTPUT_WAV}")
        print(f"Size: {size_mb:.2f} MB")
        with wave.open(OUTPUT_WAV, 'rb') as w:
            duration = w.getnframes() / w.getframerate()
            print(f"Duration: {duration:.2f} sec")
    else:
        print("❌ Final file not created!")
        sys.exit(1)

if __name__ == "__main__":
    main()
