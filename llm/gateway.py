import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed


load_dotenv()


class LLMGateway:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
            api_key=os.getenv("LLM_API_KEY", "ollama"),
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "600")),
        )
        self.model = os.getenv("LLM_MODEL", "glm4")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "256"))

    def _parse_json_content(self, content: str) -> dict:
        text = (content or "").strip()
        if not text:
            return {}

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]

        return json.loads(text)

    @retry(stop=stop_after_attempt(1), wait=wait_fixed(1))
    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return self._parse_json_content(content)
