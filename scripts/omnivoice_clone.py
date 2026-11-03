import os, shutil, wave, re
from gradio_client import Client, handle_file

REF_AUDIO = "ref_audio/sample.wav"
REF_TEXT_FILE = "ref_text.txt"
VOICE_TEXT_FILE = "voice_text.txt"
OUTPUT_DIR = "output"
OUTPUT_FILE = f"{OUTPUT_DIR}/output_voice_omnivoice.wav"
TEMP_DIR = "temp_omni"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def read_text(p):
    with open(p, 'r', encoding='utf-8') as f:
        return f.read().strip()

def merge_wav(files, out_path):
    with wave.open(files[0], 'rb') as w: params = w.getparams()
    frames = []
    for f in files:
        with wave.open(f, 'rb') as w: frames.append(w.readframes(w.getnframes()))
    with wave.open(out_path, 'wb') as out:
        out.setparams(params)
        for fr in frames: out.writeframes(fr)

ref_text = read_text(REF_TEXT_FILE)
gen_text = read_text(VOICE_TEXT_FILE)

# Chunking
sentences = re.split(r'[।\n]+', gen_text)
chunks, cur = [], ""
for s in sentences:
    s=s.strip()
    if not s: continue
    if len(cur)+len(s) < 150:
        cur += s + "। "
    else:
        if cur: chunks.append(cur.strip())
        cur = s + "। "
if cur: chunks.append(cur.strip())

print(f"Split into {len(chunks)} chunks")

# ✅ TOKEN WITH CLIENT - More quota
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print(f"Using HF_TOKEN: {hf_token[:8]}...")
    client = Client("k2-fsa/OmniVoice", hf_token=hf_token)
else:
    client = Client("k2-fsa/OmniVoice")
print("✅ Connected!")

temp_files = []
for i, chunk in enumerate(chunks):
    print(f"\n[{i+1}/{len(chunks)}] {chunk[:50]}...")
    result = client.predict(
        text=chunk, lang="Auto",
        ref_aud=handle_file(REF_AUDIO),
        ref_text=ref_text,
        instruct="male, indian accent",
        ns=32, gs=2.0, dn=True, sp=1.0, du=3, pp=True, po=True,
        api_name="/_clone_fn",
    )
    audio_path = result[0] if isinstance(result, (list, tuple)) else result
    temp_path = f"{TEMP_DIR}/part_{i}.wav"
    shutil.copy(audio_path, temp_path)
    temp_files.append(temp_path)
    print(f"✅ Chunk {i+1} done")

merge_wav(temp_files, OUTPUT_FILE)
shutil.rmtree(TEMP_DIR, ignore_errors=True)
print(f"\n✅✅✅ SUCCESS: {OUTPUT_FILE}")
