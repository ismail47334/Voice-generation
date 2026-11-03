import os
from google import genai
from google.genai import types  # সঠিক টাইপ অবজেক্ট ব্যবহারের জন্য এটি প্রয়োজন

# GitHub Secrets থেকে API Key নেওয়া
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY পাওয়া যায়নি! GitHub Secrets চেক করুন।")

# Google GenAI ক্লায়েন্ট ইনিশিয়ালাইজ করা
client = genai.Client(api_key=api_key)

# voice_text.txt ফাইল থেকে টেক্সট পড়া
input_file = "voice_text.txt"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"{input_file} ফাইলটি রিপোজিটরিতে পাওয়া যায়নি!")

with open(input_file, "r", encoding="utf-8") as f:
    text_to_speak = f.read().strip()

if not text_to_speak:
    raise ValueError(f"{input_file} ফাইলটি খালি! দয়া করে কিছু টেক্সট লিখুন।")

print("ভয়েস ডিজাইন এবং কনফিগার করা হচ্ছে...")

# নতুন SDK-এর নিয়ম অনুযায়ী SpeechConfig অবজেক্ট তৈরি করা
voice_config = types.SpeechConfig(
    voice_config=types.VoiceConfig(
        prebuilt_voice_config=types.PrebuiltVoiceConfig(
            voice_design_prompt=(
                "A Bangladeshi young male voice, speaking with a natural Bangladeshi accent. "
                "The tone must be energetic, highly excited, thrilling, and fast-paced, "
                "perfect for a cinema or movie explanation video (movie recap style)."
            )
        )
    )
)

print("ডিজাইন করা ভয়েসে বাংলা অডিও জেনারেট হচ্ছে...")

# GenerateContentConfig অবজেক্টের ভেতরে speech_config পাস করা
response = client.models.generate_content(
    model="gemini-3.8-flash-tts",
    contents=text_to_speak,
    config=types.GenerateContentConfig(
        response_mime_type="audio/mp3",
        speech_config=voice_config
    )
)

# 'output' ফোল্ডার তৈরি ও সেভ করা
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_filepath = os.path.join(output_dir, "movie_explain_speech.mp3")

with open(output_filepath, "wb") as f:
    f.write(response.candidates.content.parts.inline_data.data)

print(f"সফলভাবে অডিও তৈরি হয়েছে এবং '{output_filepath}' এ সেভ হয়েছে!")
