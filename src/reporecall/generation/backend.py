from typing import Protocol


class LLMBackend(Protocol):
    """Minimal prompt-to-text boundary used by the RAG pipeline."""

    @property
    def model_name(self) -> str:
        """Return the provider model identity used for generation."""

        ...

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate final answer text from separated instructions and evidence."""

        ...
