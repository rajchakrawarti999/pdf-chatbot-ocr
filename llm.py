import os
import time

from dotenv import load_dotenv

load_dotenv()

# Model ka naam yahan badal sakte ho (ya LLM_MODEL environment variable se)
MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")

# Answer aur summary ki language ka rule (Hinglish support)
LANG_RULE = (
    "Reply in the same language and script as the user's question. "
    "If the question is in Hinglish (Hindi written in English letters), "
    "reply in simple Hinglish."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY set nahi hai. aistudio.google.com se key lo aur "
                ".env file ya environment variable me daalo."
            )
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


def ask_llm(prompt, retries=4):
    for attempt in range(retries):
        try:
            response = _get_client().models.generate_content(model=MODEL, contents=prompt)
            return response.text or ""
        except Exception as e:
            msg = str(e)
            temporary = "503" in msg or "429" in msg or "UNAVAILABLE" in msg
            if not temporary or attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"Gemini busy hai, {wait} sec baad dobara try...")
            time.sleep(wait)