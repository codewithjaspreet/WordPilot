from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from app.llm.base import LLMProvider


class GroqProvider(LLMProvider):

    def create(self, model: str) -> BaseChatModel:
        return ChatGroq(
            model=model,
        )