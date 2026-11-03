import os
from google import genai

# GitHub Secrets থেকে API Key নেওয়া
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY পাওয়া যায়নি! GitHub Secrets চেক করুন।")

# নতুন Google GenAI ক্লায়েন্ট ইনিশিয়ালাইজ করা
client = genai.Client(api_key=api_key)

# voice_text.txt ফাইল থেকে টেক্সট পড়া
input_file = "voice_text.txt"
if not os.path.exists(input_file):
    raise FileNotFoundError(f"{input_file} ফাইলটি রিপোজিটরিতে পাওয়া যায়নি!")

with open(input_file, "r", encoding="utf-8") as f:
    text_to_speak = f.read().strip()

if not text_to_speak:
    raise ValueError(f"{input_file} ফাইলটি খালি! দয়া করে কিছু টেক্সট লিখুন।")

print("ভয়েস জেনারেট হচ্ছে...")

# নতুন SDK অনুযায়ী জেমিনি ৩.৮ ভয়েস মডেল দিয়ে অডিও তৈরি
response = client.models.generate_content(
    model="gemini-3.8-flash-tts",
    contents=text_to_speak,
    config={
        "response_mime_type": "audio/mp3",
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    # এখানে জেমিনিকে প্রম্পট দিয়ে আপনার মনের মতো ভয়েস দেওয়া হয়েছে
                    "voice_design_prompt": (
                        "A Bangladeshi young male voice, speaking with a natural Bangladeshi accent. "
                        "The tone must be energetic, highly excited, thrilling, and fast-paced, "
                        "perfect for a cinema or movie explanation video (movie recap style)."
                    )
                }
            }
        }
    }
)

# 'output' ফোল্ডার তৈরি ও সেভ করা
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_filepath = os.path.join(output_dir, "movie_explain_speech.mp3")

# নতুন লাইব্রেরির ডাটা স্ট্রাকচার অনুযায়ী অডিও ফাইল রাইট করা
with open(output_filepath, "wb") as f:
    f.write(response.candidates[0].content.parts[0].inline_data.data)

print(f"সফলভাবে অডিও তৈরি হয়েছে এবং '{output_filepath}' এ সেভ হয়েছে!")
