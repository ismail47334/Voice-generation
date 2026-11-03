import os
from google import genai
from google.genai import types

# 【entity-GitHub¦canonical_name=GitHub】 Secrets থেকে API Key নেওয়া
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY পাওয়া যায়নি! 【entity-GitHub¦canonical_name=GitHub】 Secrets চেক করুন।")

# Google GenAI ক্লায়েন্ট
client = genai.Client(api_key=api_key)

# voice_text.txt ফাইল থেকে টেক্সট পড়া
input_file = "voice_text.txt"
if not os.path.exists(input_file):
    # 【entity-GitHub¦canonical_name=GitHub】 Action এ root থেকে রান হলে path ঠিক রাখার জন্য
    if os.path.exists(f"../{input_file}"):
        input_file = f"../{input_file}"
    else:
        raise FileNotFoundError(f"{input_file} ফাইলটি রিপোজিটরিতে পাওয়া যায়নি!")

with open(input_file, "r", encoding="utf-8") as f:
    text_to_speak = f.read().strip()

if not text_to_speak:
    raise ValueError(f"{input_file} ফাইলটি খালি!")

print("ভয়েস কনফিগার করা হচ্ছে...")

# --- এটাই মেইন ফিক্স ---
# ১. prebuilt_voice_config এ voice_design_prompt দেওয়া যাবে না, শুধু voice_name দিতে হবে
# ২. Movie recap style এর জন্য Fenrir / Charon / Orus বেস্ট
# ৩. বাংলা accent এর জন্য Achernar সবচেয়ে ভালো
speech_config = types.SpeechConfig(
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(
            voice_name="Fenrir" 
        )
    )
)

print("ডিজাইন করা ভয়েসে বাংলা অডিও জেনারেট হচ্ছে...")

# নভেম্বর ২০২৬ এর লেটেস্ট ফ্রি মডেল - gemini-3.8-flash-tts
# এটা ডিসেম্বর ৩১, ২০২৬ পর্যন্ত ফ্রি
response = client.models.generate_content(
    model="gemini-3.8-flash-tts",
    contents=text_to_speak,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=speech_config
    )
)

# output ফোল্ডার তৈরি ও সেভ করা
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_filepath = os.path.join(output_dir, "movie_explain_speech.mp3")

# সঠিক ভাবে audio data বের করা
audio_data = response.candidates[0].content.parts[0].inline_data.data

with open(output_filepath, "wb") as f:
    f.write(audio_data)

print(f"সফলভাবে অডিও তৈরি হয়েছে এবং '{output_filepath}' এ সেভ হয়েছে!")
