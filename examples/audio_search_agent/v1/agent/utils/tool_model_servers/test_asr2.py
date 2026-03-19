import base64
import requests

audio_path = "/u/lmaben/speech/long_speech/temp_data/audio_0.wav"

with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "temperature": 0.01,
    "max_tokens": 512,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav"
                    }
                }
            ]
        }
    ]
}

session = requests.Session()
session.trust_env = False   # important

print("sending request...")
r = session.post(
    "http://gpua001:8000/v1/chat/completions",
    json=payload,
    timeout=300,
)
print(r.status_code)
print(r.text)