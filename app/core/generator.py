"""Step 10 of the RAG pipeline: generate the answer via the Responses API."""

from openai import OpenAI

from app.config import get_settings


class Generator:
    def __init__(self, client: OpenAI | None = None) -> None:
        settings = get_settings()
        self._client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.generation_model

    def generate(self, prompt: str) -> str:
        response = self._client.responses.create(model=self.model, input=prompt)
        return (response.output_text or "").strip()
