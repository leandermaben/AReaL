import openai

client = openai.OpenAI(
    base_url="http://gh130:8091/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    messages=[
        {"role": "user", "content": "Hello! What can you do?"}
    ]
)

print(response.choices[0].message.content)