"""OpenAI LLM client implementation."""

import os
from typing import Union, List, Dict
from openai import AsyncOpenAI
from openai import RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception_type
import logging
from .interfaces import LLMClient


def _is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a 429 rate limit error."""
    return isinstance(exception, RateLimitError)


class OpenAIGomokuClient(LLMClient):
    """OpenAI client for Gomoku gameplay."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o-mini",
        endpoint: str = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        timeout: int = 30,
        **kwargs,
    ):
        """
        Initialize OpenAI client for Gomoku gameplay.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")
            endpoint: Custom API endpoint/base URL
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            timeout: Request timeout in seconds
            **kwargs: Additional parameters for chat completion
        """
        client_kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        if endpoint:
            client_kwargs["base_url"] = endpoint
        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model

        # Generation parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.timeout = timeout
        self.extra_kwargs = kwargs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError,)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.INFO),
        reraise=True,
    )
    async def _make_api_call(self, openai_messages: List[Dict[str, str]]) -> str:
        """Make the actual API call with retry logic for rate limits."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            timeout=self.timeout,
            **self.extra_kwargs,
        )
        return response.choices[0].message.content

    async def complete(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """Send messages to OpenAI and return response."""
        try:
            # Convert input to OpenAI messages format
            if isinstance(messages, str):
                # Legacy string prompt - add default system message
                openai_messages = [
                    {
                        "role": "system",
                        "content": "You are an expert Gomoku player. Always respond with valid JSON containing your move and reasoning.",
                    },
                    {"role": "user", "content": messages},
                ]
            else:
                # Use messages directly - user has full control
                openai_messages = messages

            content = await self._make_api_call(openai_messages)

            return content

        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
