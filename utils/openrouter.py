import json
import os
import random
import re
import threading
import time
from typing import Any, Optional, Type

import httpx
from pydantic import BaseModel


def _build_response_format(schema: Type[BaseModel]) -> dict:
    raw_schema = schema.model_json_schema()

    def _patch(node: Any) -> Any:
        if isinstance(node, dict):
            result = {k: _patch(v) for k, v in node.items()}
            if result.get("type") == "object":
                result.setdefault("additionalProperties", False)
                props = result.get("properties", {})
                if props and "required" not in result:
                    result["required"] = list(props.keys())
            return result
        if isinstance(node, list):
            return [_patch(i) for i in node]
        return node

    patched = _patch(raw_schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": patched,
        },
    }


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Last resort: find outermost braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in response", text, 0)


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-oss-20b",
        timeout: float = 120.0,
        provider: Optional[dict] = None,
        reasoning: Optional[dict] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.provider = provider
        self.reasoning = reasoning

        self._thread_local = threading.local()
        self._all_clients: list[httpx.Client] = []
        self._clients_lock = threading.Lock()

    def _get_http_client(self) -> httpx.Client:
        if not hasattr(self._thread_local, "client"):
            client = httpx.Client(timeout=self.timeout)
            self._thread_local.client = client
            with self._clients_lock:
                self._all_clients.append(client)
        return self._thread_local.client

    def _request_with_backoff(self, payload: dict) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        max_attempts = 5
        response: Optional[httpx.Response] = None

        for attempt in range(max_attempts):
            try:
                response = self._get_http_client().post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 429:
                    try:
                        wait = int(response.headers.get("retry-after", 60))
                    except (ValueError, TypeError):
                        wait = 60
                    wait += random.uniform(0, 5)
                    time.sleep(wait)
                    continue

                if response.is_error:
                    raise httpx.HTTPStatusError(
                        f"{response.status_code} {response.reason_phrase}: {response.text}",
                        request=response.request,
                        response=response,
                    )

                return response

            except httpx.RequestError as exc:
                if attempt == max_attempts - 1:
                    raise exc
                time.sleep(1 + random.uniform(0, 1))

        if response is not None:
            raise httpx.HTTPStatusError(
                f"Rate limit exceeded after {max_attempts} attempts",
                request=response.request,
                response=response,
            )

        raise RuntimeError(f"Request failed after {max_attempts} attempts.")

    def generate(
        self,
        prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        model = model or self.model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if output_schema is not None:
            payload["response_format"] = _build_response_format(output_schema)

        if self.provider is not None:
            payload["provider"] = self.provider
        else:
            payload["provider"] = {"sort": "throughput"}
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if max_tokens:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        response = self._request_with_backoff(payload)
        content = response.json()["choices"][0]["message"]["content"]

        if output_schema is not None:
            try:
                return output_schema.model_validate(_parse_json(content))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse structured output: {exc}\nRaw response: {content}"
                )

        return content

    def close(self):
        with self._clients_lock:
            for client in self._all_clients:
                client.close()
            self._all_clients.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def generate(
    prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    model: str = "google/gemini-2.0-flash-001",
    api_key: Optional[str] = None,
    **kwargs: Any,
):
    with OpenRouterClient(api_key=api_key, model=model) as client:
        return client.generate(prompt, output_schema=output_schema, **kwargs)
