import os
import google.generativeai as genai
from google.generativeai import types

# GitHub Secrets থেকে API Key নেওয়া
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY পাওয়া যায়নি! GitHub Secrets চেক করুন।")

# Gemini API কনফিগারেশন
genai.configure(api_key=api_key)

# voice_text.txt ফাইল থেকে টেক্সট পড়া
input_file = "voice_text.txt"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"{input_file} ফাইলটি রিপোজিটরিতে পাওয়া যায়নি!")

with open(input_file, "r", encoding="utf-8") as f:
    text_to_speak = f.read().strip()

if not text_to_speak:
    raise ValueError(f"{input_file} ফাইলটি খালি! দয়া করে কিছু টেক্সট লিখুন।")

print("ভয়েস ডিজাইন এবং কনফিগার করা হচ্ছে...")

# আপনার পছন্দমতো ভয়েস ডিজাইন (Male, Bangladeshi Accent, Excited Movie Explainer)
voice_config = types.SpeechConfig(
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(
            # এখানে জেমিনিকে প্রম্পট দিয়ে ভয়েস ডিজাইন করা হচ্ছে
            voice_design_prompt=(
                "A Bangladeshi young male voice, speaking with a natural Bangladeshi accent. "
                "The tone must be energetic, highly excited, thrilling, and fast-paced, "
                "perfect for a cinema or movie explanation video (movie recap style)."
            )
        )
    )
)

# জেমিনি মডেল ক্লায়েন্ট তৈরি (TTS মডেলের জন্য)
# স্পিচ কনফিগারেশনসহ মডেল কল করা
model = genai.GenerativeModel(
    model_name="gemini-3.8-flash-tts",
    generation_config=genai.GenerationConfig(
        response_mime_type="audio/mp3"
    )
)

print("ডিজাইন করা ভয়েসে বাংলা অডিও জেনারেট হচ্ছে...")

# ভয়েস কনফিগারেশন পাস করে কন্টেন্ট জেনারেট করা
response = model.generate_content(
    text_to_speak,
    generation_config={"speech_config": voice_config}
)

# 'output' ফোল্ডার তৈরি এবং সেভ করা
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_filepath = os.path.join(output_dir, "movie_explain_speech.mp3")
with open(output_filepath, "wb") as f:
    f.write(response.candidates.content.parts.inline_data.data)

print(f"সফলভাবে মুভি এক্সপ্লেইন স্টাইলের অডিও তৈরি হয়েছে: '{output_filepath}'")
