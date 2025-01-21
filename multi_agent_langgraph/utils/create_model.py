from langchain_openai import ChatOpenAI

from ..types import BaseChatModel


def create_llm_model() -> BaseChatModel:
    return ChatOpenAI(name="gpt-4o")
