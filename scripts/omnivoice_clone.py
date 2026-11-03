import os, sys, shutil

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

print("=== OmniVoice Bengali Voice Clone Test - FIXED ===")

ref_text = read_text(REF_TEXT_FILE)
gen_text = read_text(VOICE_TEXT_FILE)

print(f"Ref: {ref_text[:100]}")
print(f"Gen ({len(gen_text)} chars): {gen_text[:150]}...")

from gradio_client import Client, handle_file

print("\nConnecting to k2-fsa/OmniVoice...")
client = Client("k2-fsa/OmniVoice")
print("✅ Connected!")

# FIXED: instruct must be from allowed list
# For Bengali male voice: "male, indian accent"
result = client.predict(
    text=gen_text,
    lang="Auto",
    ref_aud=handle_file(REF_AUDIO),
    ref_text=ref_text,
    instruct="male, indian accent",  # ✅ FIXED - was free text before
    ns=32,
    gs=2.0,
    dn=True,
    sp=1.0,
    du=3,
    pp=True,
    po=True,
    api_name="/_clone_fn",
)

print(f"\nResult: {result}")

# Handle result
audio_path = result[0] if isinstance(result, (list, tuple)) else result

if audio_path and os.path.exists(audio_path):
    shutil.copy(audio_path, OUTPUT_FILE)
    print(f"\n✅✅✅ SUCCESS: {OUTPUT_FILE}")
    print(f"Size: {os.path.getsize(OUTPUT_FILE)/1024/1024:.2f} MB")
else:
    print(f"❌ Failed, result: {result}")
    sys.exit(1)
