
import streamlit as st
import ollama
from google import genai
from google.genai import types

OLLAMA_MODEL = "qwen3:4b"
GEMINI_MODEL = "gemini-3.8-flash"


def generate_ai_text(prompt, system_prompt="", max_tokens=500):
    # Local mode is the default.
    provider = st.secrets.get("AI_PROVIDER", "ollama").lower()

    if provider == "gemini":
        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from Streamlit Secrets.")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt or None,
                temperature=0.1,
                max_output_tokens=max_tokens
            )
        )

        result = (response.text or "").strip()

        if not result:
            raise ValueError("Gemini returned an empty response.")

        return result

    if provider == "ollama":
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": 0.1,
                "num_predict": max_tokens
            },
            think=False
        )

        return response["message"]["content"].strip()

    raise ValueError(
        f"Unsupported AI provider: {provider}. Use 'ollama' or 'gemini'."
    )
