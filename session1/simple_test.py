import os

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model=os.environ["TINYAGENT_MODEL"],
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)

print(response.choices[0].message.content)
