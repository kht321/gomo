"""LLM-specific interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Union


class LLMClient(ABC):
    """Abstract interface for LLM clients."""

    @abstractmethod
    async def complete(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """
        Send messages to LLM and return response.

        Args:
            messages: Either a simple string prompt (legacy) or a list of message dicts
                     Each message dict should have 'role' and 'content' keys:
                     - role: 'system', 'user', or 'assistant'
                     - content: The message content

        Returns:
            str: The LLM response

        Examples:
            # Simple prompt (legacy)
            await client.complete("What's the weather?")

            # Messages format (recommended)
            await client.complete([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather?"}
            ])
        """
        pass
