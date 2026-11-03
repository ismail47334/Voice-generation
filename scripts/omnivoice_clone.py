import os
import sys
import shutil

# Files from your repo
REF_AUDIO = "ref_audio/sample.wav"
REF_TEXT_FILE = "ref_text.txt"
VOICE_TEXT_FILE = "voice_text.txt"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/output_voice_omnivoice.wav"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_text(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

print("=== OmniVoice Bengali Voice Clone Test ===")

ref_text = read_text(REF_TEXT_FILE)
gen_text = read_text(VOICE_TEXT_FILE)

if not os.path.exists(REF_AUDIO):
    print(f"❌ {REF_AUDIO} not found!")
    sys.exit(1)
if not gen_text:
    print(f"❌ {VOICE_TEXT_FILE} empty!")
    sys.exit(1)

print(f"Ref Audio: {REF_AUDIO}")
print(f"Ref Text: {ref_text[:100]}")
print(f"Gen Text ({len(gen_text)} chars): {gen_text[:150]}...")

try:
    print("\n--- Installing & Connecting to k2-fsa/OmniVoice ---")
    from gradio_client import Client, handle_file

    print("Connecting to HF Space (may take 20-30s if space is sleeping)...")
    client = Client("k2-fsa/OmniVoice")
    print("✅ Connected!")

    print("\n--- Generating (Queue may take 30s - 2min) ---")
    result = client.predict(
        text=gen_text,
        lang="Auto",  # Auto detects Bengali
        ref_aud=handle_file(REF_AUDIO),
        ref_text=ref_text,
        instruct="Speak in Bengali with natural emotion and clear pronunciation",
        ns=32,
        gs=2.0,
        dn=True,
        sp=1.0,
        du=3,
        pp=True,
        po=True,
        api_name="/_clone_fn",
    )

    print(f"\nRaw result: {result}")
    print(f"Result type: {type(result)}")

    # Result can be a filepath or tuple
    audio_path = None
    if isinstance(result, str) and os.path.exists(result):
        audio_path = result
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        # Sometimes returns (audio_path, ...)
        for item in result:
            if isinstance(item, str) and os.path.exists(item):
                audio_path = item
                break
            if isinstance(item, dict) and 'value' in item:
                audio_path = item['value']
                break

    if not audio_path:
        # If result is path string but not exists locally (gradio temp file), try to use directly
        audio_path = result if isinstance(result, str) else str(result)

    print(f"Audio path: {audio_path}")

    if os.path.exists(audio_path):
        shutil.copy(audio_path, OUTPUT_FILE)
        size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
        print(f"\n✅✅✅ SUCCESS!")
        print(f"File: {OUTPUT_FILE}")
        print(f"Size: {size_mb:.2f} MB")
    else:
        print(f"❌ File not found at {audio_path}, trying to download...")
        # If it's a remote URL, gradio_client already downloaded to temp
        # Just try to copy whatever result gave
        if isinstance(result, str):
            import requests
            if result.startswith("http"):
                print(f"Downloading from URL: {result}")
                r = requests.get(result, stream=True)
                with open(OUTPUT_FILE, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                print(f"✅ Downloaded to {OUTPUT_FILE}")
            else:
                print(f"❌ Unexpected result format: {result}")
                sys.exit(1)

except Exception as e:
    print(f"\n❌ OmniVoice API failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
