from __future__ import annotations

import time

import httpx

from news_pipeline.config import get_settings
from news_pipeline.contracts import LLMResponse
from news_pipeline.llm.rate_limit import RequestRateLimiter, get_shared_rate_limiter
from news_pipeline.llm.provider import LLMProvider, LLMProviderError, LLMTraceContext
from news_pipeline.tracking.experiment import log_llm_call_trace


class GroqProvider(LLMProvider):
    provider_name = "groq"

    def __init__(
        self,
        client: httpx.Client | None = None,
        rate_limiter: RequestRateLimiter | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model
        self.base_url = settings.groq_base_url.rstrip("/")
        self.max_retries = settings.llm_max_retries
        self.rate_limiter = rate_limiter or get_shared_rate_limiter(
            self.provider_name,
            settings.llm_requests_per_minute,
            backend=settings.llm_rate_limit_backend,
            database_url=settings.database_url,
        )
        self.client = client or httpx.Client(
            timeout=settings.llm_timeout_seconds,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def complete(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 900,
        trace_context: LLMTraceContext | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMProviderError("GROQ_API_KEY is not configured")
        return self._complete_with_retry(prompt, system_prompt, temperature, max_tokens, trace_context)

    def _complete_with_retry(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        trace_context: LLMTraceContext | None,
    ) -> LLMResponse:
        settings = get_settings()
        last_error: Exception | None = None
        attempts: list[dict[str, object]] = []
        request_payload = {
            "provider_name": self.provider_name,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "prompt": prompt,
        }

        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.acquire()
            started = time.perf_counter()
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": {"type": "json_object"},
                    },
                )
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response.headers.get("retry-after"))
                    message = "Groq rate limited the request"
                    if retry_after is not None:
                        message = f"{message}; retry_after={self._format_retry_after(retry_after)}"
                        self.rate_limiter.backoff(retry_after)
                    last_error = LLMProviderError(message)
                    attempts.append(
                        self._attempt_record(
                            attempt=attempt,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            status_code=response.status_code,
                            error_message=message,
                            retry_after=retry_after,
                        )
                    )
                    if attempt >= self.max_retries:
                        self._log_trace(
                            settings.mlflow_experiment_extraction,
                            trace_context,
                            request_payload,
                            attempts,
                            None,
                            str(last_error),
                        )
                        raise last_error
                    time.sleep(self._retry_delay(attempt, retry_after))
                    continue

                if response.status_code >= 500:
                    last_error = LLMProviderError(f"Groq server error: {response.status_code}")
                    attempts.append(
                        self._attempt_record(
                            attempt=attempt,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            status_code=response.status_code,
                            error_message=str(last_error),
                        )
                    )
                    if attempt >= self.max_retries:
                        self._log_trace(
                            settings.mlflow_experiment_extraction,
                            trace_context,
                            request_payload,
                            attempts,
                            None,
                            str(last_error),
                        )
                        raise last_error
                    time.sleep(self._default_backoff(attempt))
                    continue

                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = LLMProviderError(
                    f"Client error '{error.response.status_code} {error.response.reason_phrase}' for url '{error.request.url}'"
                )
                attempts.append(
                    self._attempt_record(
                        attempt=attempt,
                        latency_ms=latency_ms,
                        status_code=error.response.status_code,
                        error_message=str(last_error),
                    )
                )
                self._log_trace(
                    settings.mlflow_experiment_extraction,
                    trace_context,
                    request_payload,
                    attempts,
                    None,
                    str(last_error),
                )
                raise last_error from error
            except httpx.HTTPError as error:
                latency_ms = int((time.perf_counter() - started) * 1000)
                last_error = error
                attempts.append(
                    self._attempt_record(
                        attempt=attempt,
                        latency_ms=latency_ms,
                        error_message=str(error),
                    )
                )
                if attempt >= self.max_retries:
                    self._log_trace(
                        settings.mlflow_experiment_extraction,
                        trace_context,
                        request_payload,
                        attempts,
                        None,
                        str(error),
                    )
                    raise
                time.sleep(self._default_backoff(attempt))
                continue

            payload = response.json()
            latency_ms = int((time.perf_counter() - started) * 1000)
            llm_response = LLMResponse(
                text=payload["choices"][0]["message"]["content"],
                model=payload.get("model", self.model),
                tokens_used=payload.get("usage", {}).get("total_tokens", 0),
                latency_ms=latency_ms,
                provider_name=self.provider_name,
            )
            attempts.append(
                self._attempt_record(
                    attempt=attempt,
                    latency_ms=latency_ms,
                    status_code=response.status_code,
                )
            )
            self._log_trace(
                settings.mlflow_experiment_extraction,
                trace_context,
                request_payload,
                attempts,
                {
                    "text": llm_response.text,
                    "model": llm_response.model,
                    "tokens_used": llm_response.tokens_used,
                    "latency_ms": llm_response.latency_ms,
                    "provider_name": llm_response.provider_name,
                },
                None,
            )
            return llm_response

        if last_error is not None:
            self._log_trace(
                settings.mlflow_experiment_extraction,
                trace_context,
                request_payload,
                attempts,
                None,
                str(last_error),
            )
            raise last_error
        raise LLMProviderError("Groq request failed without a response")

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        return max(parsed, 0.0)

    @staticmethod
    def _format_retry_after(value: float) -> str:
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"

    @staticmethod
    def _default_backoff(attempt: int) -> float:
        return min(2 ** (attempt - 1), 8)

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        return self._default_backoff(attempt)

    @staticmethod
    def _attempt_record(
        *,
        attempt: int,
        latency_ms: int,
        status_code: int | None = None,
        error_message: str | None = None,
        retry_after: float | None = None,
    ) -> dict[str, object]:
        record: dict[str, object] = {"attempt": attempt, "latency_ms": latency_ms}
        if status_code is not None:
            record["status_code"] = status_code
        if error_message is not None:
            record["error_message"] = error_message
        if retry_after is not None:
            record["retry_after"] = retry_after
        return record

    def _log_trace(
        self,
        experiment_name: str,
        trace_context: LLMTraceContext | None,
        request_payload: dict[str, object],
        attempts: list[dict[str, object]],
        response_payload: dict[str, object] | None,
        error_message: str | None,
    ) -> None:
        log_llm_call_trace(
            experiment_name=experiment_name,
            provider_name=self.provider_name,
            model_name=self.model,
            trace_context=trace_context.to_dict() if trace_context is not None else None,
            request_payload=request_payload,
            attempts=attempts,
            response_payload=response_payload,
            error_message=error_message,
        )
