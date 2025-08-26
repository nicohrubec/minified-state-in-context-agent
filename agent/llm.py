import time
from openai import OpenAI


client = OpenAI()


def call_gpt(system_prompt, user_prompt, model="gpt-4.1"):
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
                reasoning_effort="medium" if "gpt-5" in model else None,
                temperature=0.2 if "gpt-5" not in model else 1.0,
                model=model,
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
