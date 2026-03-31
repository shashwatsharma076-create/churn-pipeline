"""
Base agent class for all AI agents.
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Dict
from openai import OpenAI
from dotenv import load_dotenv
from src.config import API_CONFIG

load_dotenv()


class BaseAgent(ABC):
    """Base class for all agents in the pipeline."""

    def __init__(self, model: str = None, temperature: float = None):
        self.client = OpenAI(
            base_url=API_CONFIG["base_url"],
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = model or API_CONFIG["default_model"]
        self.temperature = temperature or API_CONFIG["temperature"]
        self.max_tokens = API_CONFIG["max_tokens"]

    def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Make a call to the LLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    @abstractmethod
    def run(self, data: Any) -> Any:
        """Run the agent on the given data."""
        pass
