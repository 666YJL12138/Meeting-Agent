import ast
import json
import logging
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Any


load_dotenv()

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _build_client(
    base_url: str,
    api_key: str,
    timeout: float,
) -> OpenAI:
    """Reuse one HTTP client per endpoint within the API process."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


class LLMGateway:
    def __init__(self):
        self.base_url = os.getenv(
            "LLM_BASE_URL",
            "http://127.0.0.1:11434/v1",
        )
        self.api_key = os.getenv("LLM_API_KEY", "ollama")
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "900"))
        self.client = _build_client(
            self.base_url,
            self.api_key,
            self.timeout,
        )

        self.model = os.getenv("LLM_MODEL", "glm4")
        self.temperature = float(
            os.getenv("LLM_TEMPERATURE", "0")
        )
        self.max_tokens = int(
            os.getenv("LLM_MAX_TOKENS", "1536")
        )

        self.use_json_format = os.getenv(
            "LLM_USE_JSON_FORMAT",
            "1",
        ) == "1"
        self.keep_alive = os.getenv("LLM_KEEP_ALIVE", "10m")
        self.num_ctx = int(os.getenv("LLM_NUM_CTX", "4096"))
        self.num_predict = int(
            os.getenv("LLM_NUM_PREDICT", str(self.max_tokens))
        )
        self.use_ollama_options = os.getenv(
            "LLM_USE_OLLAMA_OPTIONS",
            "1",
        ) == "1"

    def _clean_model_content(self, content: str) -> str:
        """清理 BOM、Markdown 代码块和模型前后的解释文字。"""
        text = (content or "").replace("\ufeff", "").strip()

        fenced = re.search(
            r"```(?:json|python|text)?\s*(.*?)\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if fenced:
            return fenced.group(1).strip()

        return text

    def _extract_json_object(self, content: str) -> str:
        """
        从模型返回文本中提取完整 JSON 对象。
        可以处理：
        1. ```json 或 ```python 代码块
        2. JSON 前面的解释文字
        3. JSON 后面的解释文字
        4. JSON 字符串内部的大括号
        """

        text = self._clean_model_content(content)

        if not text:
            raise ValueError("LLM returned empty content")

        starts = [
            index
            for index, char in enumerate(text)
            if char in "[{"
        ]

        if not starts:
            raise ValueError(
                f"LLM response does not contain JSON: {text[:500]}"
            )

        decoder = json.JSONDecoder()

        for start in starts:
            try:
                _, end = decoder.raw_decode(text[start:])
                return text[start:start + end]
            except json.JSONDecodeError:
                # 兼容 Python 风格字面量，例如：
                # json.dumps({"ok": True, "message": "模型正常"})
                balanced = self._extract_balanced_value(text, start)
                if balanced:
                    return balanced

        preview = text[:1500].replace("\n", "\\n")

        raise ValueError(
            "LLM returned no valid JSON object. "
            f"preview={preview}"
        )

    def _extract_balanced_value(
        self,
        text: str,
        start: int,
    ) -> str | None:
        """提取括号完整的对象/数组，供 Python 字面量兜底解析。"""
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        quote: str | None = None
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue

            if char in ("'", '"'):
                quote = char
                continue

            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        return None

    def _parse_json_content(self, content: str) -> dict:
        cleaned_content = self._clean_model_content(content)
        json_text = self._extract_json_object(cleaned_content)

        try:
            result = json.loads(json_text)
        except json.JSONDecodeError as exc:
            # 某些本地模型会返回 json.dumps({"ok": True})，
            # 其中 True/False/None 是 Python 字面量，不是 JSON。
            # 这里只使用 ast.literal_eval，不执行任意代码。
            try:
                result = ast.literal_eval(json_text)
            except (ValueError, SyntaxError):
                preview = json_text[:1500].replace("\n", "\\n")

                raise ValueError(
                    f"LLM returned invalid JSON: {preview}"
                ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "LLM JSON result must be an object"
            )

        return result

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
        reraise=True,
    )
    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        request = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }

        if self.use_json_format:
            request["response_format"] = {
                "type": "json_object"
            }

        if self.use_ollama_options:
            request["extra_body"] = {
                "keep_alive": self.keep_alive,
                "options": {
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            }

        response = self.client.chat.completions.create(
            **request
        )

        content = (
            response.choices[0].message.content
            or ""
        )

        try:
            return self._parse_json_content(content)

        except Exception as exc:
            preview = content[:1500].replace(
                "\n",
                "\\n",
            )

            logger.error(
                "LLM JSON parse failed. "
                "model=%s preview=%s error=%s",
                self.model,
                preview,
                exc,
            )

            raise
