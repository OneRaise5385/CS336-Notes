from sqlitedict import SqliteDict
import os
from file_util import ensure_directory_exists


def query_model(model: str, prompt: str) -> str:
    """Query `model` with the `prompt` and return the top-1 response."""
    ensure_directory_exists("var")
    cache = SqliteDict("var/query_model_cache.db")
    key = model + ":" + prompt
    if key in cache:
        return cache[key]

    from openai import OpenAI
    if model.startswith("gpt-"):
        # Use an actual OpenAI model
        client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
        )
    else:
        # Together API serves open models using the same OpenAI interface
        client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )

    system_prompt = "You are a helpful and harmless assistant."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    value = response.choices[0].message.content
    cache[key] = value
    cache.commit()

    return value


def query_gpt4o(prompt: str) -> str:
    return query_model(model="gpt-4o", prompt=prompt)


def query_deepseek_v3(prompt: str) -> str:
    return query_model(model="deepseek-v3", prompt=prompt)

