from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_ANSWER_MODEL = "gpt-5-mini"


@dataclass(frozen=True)
class EmbeddingResponse:
    model: str
    embeddings: list[list[float]]


@dataclass(frozen=True)
class TextResponse:
    model: str
    text: str


def load_dotenv(dotenv_path: Path | None = None) -> None:
    path = dotenv_path or Path.cwd() / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class OpenAIBaseClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        load_dotenv()
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The 'openai' package is required for OpenAI commands.") from exc

        client_kwargs = {"api_key": resolved_api_key}
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        resolved_organization = organization or os.getenv("OPENAI_ORG_ID")
        resolved_project = project or os.getenv("OPENAI_PROJECT_ID")

        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        if resolved_organization:
            client_kwargs["organization"] = resolved_organization
        if resolved_project:
            client_kwargs["project"] = resolved_project

        self._client = OpenAI(**client_kwargs)


class OpenAIEmbeddingClient(OpenAIBaseClient):
    def create_embeddings(
        self,
        texts: Iterable[str],
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
    ) -> EmbeddingResponse:
        inputs = list(texts)
        if not inputs:
            return EmbeddingResponse(model=model, embeddings=[])

        request_args: dict[str, object] = {
            "model": model,
            "input": inputs,
            "encoding_format": "float",
        }
        if dimensions is not None:
            request_args["dimensions"] = dimensions

        response = self._client.embeddings.create(**request_args)
        embeddings = [item.embedding for item in response.data]
        return EmbeddingResponse(model=model, embeddings=embeddings)


class OpenAIAnswerClient(OpenAIBaseClient):
    def stream_answer(
        self,
        prompt: str,
        model: str = DEFAULT_ANSWER_MODEL,
        temperature: float | None = None,
    ) -> Iterator[str]:
        yield from self._stream_with_responses_api(prompt=prompt, model=model, temperature=temperature)

    def create_answer(
        self,
        prompt: str,
        model: str = DEFAULT_ANSWER_MODEL,
        temperature: float | None = None,
    ) -> TextResponse:
        text = self._create_with_responses_api(prompt=prompt, model=model, temperature=temperature)
        return TextResponse(model=model, text=text.strip())

    def _create_with_responses_api(self, prompt: str, model: str, temperature: float | None) -> str:
        if not hasattr(self._client, "responses"):
            return self._fallback_chat_completions(prompt=prompt, model=model, temperature=temperature)

        request_args = self._build_response_request(prompt=prompt, model=model, temperature=temperature)

        try:
            response = self._client.responses.create(**request_args)
        except Exception:
            return self._fallback_chat_completions(prompt=prompt, model=model, temperature=temperature)

        text = getattr(response, "output_text", None)
        if text:
            return str(text)
        extracted = self._extract_output_text(response)
        if extracted:
            return extracted
        return self._fallback_chat_completions(prompt=prompt, model=model, temperature=temperature)

    def _fallback_chat_completions(self, prompt: str, model: str, temperature: float | None) -> str:
        request_args: dict[str, object] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant. Answer strictly from the provided context and cite the source ids you used.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if temperature is not None:
            request_args["temperature"] = temperature

        response = self._client.chat.completions.create(**request_args)
        return response.choices[0].message.content or ""

    def _extract_output_text(self, response: object) -> str:
        pieces: list[str] = []
        for output in getattr(response, "output", []) or []:
            for content in getattr(output, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    pieces.append(str(text))
        return "\n".join(pieces)

    def _stream_with_responses_api(self, prompt: str, model: str, temperature: float | None) -> Iterator[str]:
        if not hasattr(self._client, "responses") or not hasattr(self._client.responses, "stream"):
            yield self._create_with_responses_api(prompt=prompt, model=model, temperature=temperature)
            return

        request_args = self._build_response_request(prompt=prompt, model=model, temperature=temperature)
        try:
            with self._client.responses.stream(**request_args) as stream:
                emitted = False
                for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            emitted = True
                            yield str(delta)
                if not emitted:
                    response = stream.get_final_response()
                    text = getattr(response, "output_text", None) or self._extract_output_text(response)
                    if text:
                        yield str(text)
        except Exception:
            yield self._create_with_responses_api(prompt=prompt, model=model, temperature=temperature)

    def _build_response_request(self, prompt: str, model: str, temperature: float | None) -> dict[str, object]:
        request_args: dict[str, object] = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "You are a research assistant. Answer strictly from the provided context and cite the source ids you used.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                },
            ],
        }
        if temperature is not None:
            request_args["temperature"] = temperature
        return request_args
