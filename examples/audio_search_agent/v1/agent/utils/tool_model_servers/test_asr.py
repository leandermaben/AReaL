import base64
import os
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://gpua001:8000/v1",
    api_key="EMPTY",
)

audio_path = "/u/lmaben/speech/long_speech/temp_data/audio_0.wav"

print("starting")
print("file exists:", os.path.exists(audio_path))
print("file size MB:", os.path.getsize(audio_path) / (1024 * 1024))

t0 = time.time()
with open(audio_path, "rb") as f:
    audio_bytes = f.read()
print("read done in", time.time() - t0, "sec")

t1 = time.time()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
print("base64 done in", time.time() - t1, "sec")
print("base64 length:", len(audio_b64))

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav",
                },
            }
        ],
    }
]

print("sending request...")
t2 = time.time()
resp = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    temperature=0.01,
    max_tokens=512,
    messages=messages,
    timeout=300,
)
print("request finished in", time.time() - t2, "sec")

content = resp.choices[0].message.content
print(content)