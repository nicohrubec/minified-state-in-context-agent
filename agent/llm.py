import os
import time
from openai import AzureOpenAI


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
deployment = "gpt-4.1"


def call_gpt(system_prompt, user_prompt):
    retry_delay = 60  # seconds
    attempt = 1
    max_attempts = 10

    while True:
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=32768,
                temperature=0.2,
                model=deployment,
            )
            return response.choices[0].message.content

        except Exception as e:
            if attempt > max_attempts:
                raise e
            print(
                f"ChatGPT request failed {e}. Retrying in {retry_delay} seconds... (Attempt {attempt})"
            )
            time.sleep(retry_delay)
            attempt += 1
            continue
