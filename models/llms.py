
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib import request

import anthropic
import torch
from openai import OpenAI
from mistralai.client import Mistral
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types as genai_types

from uilts.configs import Configs
from uilts.logger import logger
from uilts.configs import LLMConfigs


class BaseLLM(LLMConfigs):
    """Common interface for provider-specific LLM callers (Claude, OpenAI, Google, Grok, Ollama, Mistral, HuggingFace, ...).

    Subclasses implement two methods:
      - `_initialize_model(**kwargs)`: build the provider's SDK client and assign it to `self.model_client`.
        Any kwarg matching an existing class attribute (e.g. `top_p`, `system`) is applied via `setattr`,
        so provider-specific options can be set at construction time.
      - `_call(prompt, **kwargs)`: run one generation against that client and return plain text.

    Provider-specific optional parameters are declared as class attributes with defaults, and their names
    are listed in a class-level `arguments` list. `_call` walks that list and forwards each one into the
    API request, preferring a per-call `kwargs` value over the instance default set at construction. This
    lets every param be configured once and still be overridden per-call without changing the method
    signature.

    Args:
        model_name: Provider/model identifier (e.g. "claude-sonnet-5", "gpt-4o").
        temperature: Sampling temperature; overridable per-call via kwargs.
        max_tokens: Max tokens to generate; overridable per-call via kwargs.
        api_key: Provider API key used to construct the client.
        tools: Tool/function-calling schema list, in the calling provider's expected format.
        type: Role this LLM plays ("generator", "retriever", "tool caller", "tool generator").
        mcp_servers: Remote MCP servers to connect for this call (currently used by ClaudeLLM only).
        **kwargs: Provider-specific overrides (e.g. `top_p`, `system`, `stream`) forwarded to
            `_initialize_model` and applied to matching class attributes on the subclass.
    """
    model_client: InferenceClient | anthropic.Anthropic | OpenAI | Mistral | genai.Client = None
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        tools: list[dict[str, str]],
        type: str,
        mcp_servers: list[str] | None = None,  # remote MCP servers to connect for this call
        **kwargs: Any
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            tools=tools,
            type=type,
            mcp_servers=mcp_servers
        )
        self.api_key_checker()
        self._initialize_model(**kwargs)

    def api_key_checker(self) -> str:
        """Resolve `api_key` as either an environment variable name or a literal key value.

        Tries `api_key` as an environment variable name first; if that's unset, falls back to
        treating `api_key` itself as the literal key. Raises if neither is available.
        """
        if os.getenv(self.api_key):
            self.api_key = os.getenv(self.api_key)

        if not self.api_key:
            logger.error(f"No API key or environment variable name provided for {self.model_name}.")
            raise ValueError(
                f"Missing API key for {self.model_name}: provide a direct key or an environment variable name."
            )

    @abstractmethod
    def _initialize_model(self, **kwargs: Any) -> None:
        """Initialize provider-specific model/client state."""

    def _call(self, prompt: str, **kwargs: Any) -> str:
        """Run one generation call and return plain text."""


class HuggingFaceInferenceLLM(BaseLLM):
    """Free-tier Hugging Face Inference API caller, via `huggingface_hub.InferenceClient` (OpenAI-compatible).

    Extra params: `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `stream`, `tool_choice`, `seed`.
    """
    top_p: float = 1.0  # nucleus sampling threshold
    frequency_penalty: float = 0.0  # penalize tokens by how often they've already appeared
    presence_penalty: float = 0.0  # penalize tokens that have appeared at all so far
    stop: list[str] | None = None  # strings that halt generation early
    stream: bool = False  # stream the response instead of waiting for the full completion
    tool_choice: str | dict[str, Any] | None = None  # controls how/whether tools are invoked
    seed: int | None = None  # deterministic sampling seed
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'top_p',
        'frequency_penalty',
        'presence_penalty',
        'stop',
        'stream',
        'tool_choice',
        'seed'
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = InferenceClient(api_key=self.api_key, provider="featherless-ai")

    def _call(self, prompt: str, **kwargs: Any) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "messages": [{"role": "user", "content": str(prompt)}],
            "tools": kwargs.get("tools", self.tools),
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs.get(args, getattr(self, args, None))

        response = self.model_client.chat_completion(**call_kwargs)
        logger.info(f"HF Inference API response: {response}")
        return str(response.choices[0].message.content or "").strip()


class ClaudeLLM(BaseLLM):
    """Anthropic Claude API caller, via the `anthropic` SDK's `messages.create`.

    Extra params: `system`, `top_p`, `top_k`, `stop_sequences`, `stream`, `tool_choice`, `metadata`,
    `thinking`, `service_tier`, `container`. Also the only caller that forwards `mcp_servers`.
    """
    system: str | None = None  # system prompt / persona instructions
    top_p: float = 1.0  # nucleus sampling threshold
    top_k: int = 1  # restrict sampling to the top K candidate tokens
    stop_sequences: list[str] | None = None  # strings that halt generation early
    stream: bool = False  # stream the response instead of waiting for the full completion
    tool_choice: dict[str, Any] | None = None  # controls how/whether tools are invoked
    metadata: dict[str, str] | None = None  # request metadata, e.g. user_id for abuse tracking
    thinking: dict[str, Any] | None = None  # extended thinking / reasoning budget config
    service_tier: str | None = None  # request priority routing tier
    container: str | None = None  # sandbox/container id for the code execution tool
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'system', 
        'top_p', 
        'top_k', 
        'stop_sequences', 
        'stream', 
        'tool_choice', 
        'metadata', 
        'thinking', 
        'service_tier', 
        'container'
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = anthropic.Anthropic(api_key=self.api_key)

    def _call(self, prompt: str, **kwargs: Any) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "messages": [{"role": "user", "content": prompt}],
            "tools": kwargs.get("tools", self.tools),
            "mcp_servers": kwargs.get("mcp_servers", self.mcp_servers)
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs[args]

        response = self.model_client.messages.create(**call_kwargs)
        logger.info(f"Claude API response: {response}")
        return "".join(block.text for block in response.content if block.type == "text").strip()


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completions API caller, via the `openai` SDK's `chat.completions.create`.

    Extra params: `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `stream`, `tool_choice`,
    `seed`, `response_format`, `user`, `service_tier`.
    """
    top_p: float = 1.0  # nucleus sampling threshold
    frequency_penalty: float = 0.0  # penalize tokens by how often they've already appeared
    presence_penalty: float = 0.0  # penalize tokens that have appeared at all so far
    stop: list[str] | None = None  # strings that halt generation early
    stream: bool = False  # stream the response instead of waiting for the full completion
    tool_choice: str | dict[str, Any] | None = None  # controls how/whether tools are invoked
    seed: int | None = None  # deterministic sampling seed
    response_format: dict[str, Any] | None = None  # force structured output, e.g. JSON mode
    user: str | None = None  # end-user identifier for abuse tracking
    service_tier: str | None = None  # request priority routing tier
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'top_p',
        'frequency_penalty',
        'presence_penalty',
        'stop',
        'stream',
        'tool_choice',
        'seed',
        'response_format',
        'user',
        'service_tier'
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = OpenAI(api_key=self.api_key)

    def _call(self, prompt: str, **kwargs: Any) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "messages": [{"role": "user", "content": prompt}],
            "tools": kwargs.get("tools", self.tools),
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs[args]

        response = self.model_client.chat.completions.create(**call_kwargs)
        logger.info(f"OpenAI API response: {response}")
        return str(response.choices[0].message.content or "").strip()


class GoogleLLM(BaseLLM):
    """Google Gemini API caller, via the `google-genai` SDK's `models.generate_content`.

    Extra params (passed through `GenerateContentConfig`): `system_instruction`, `top_p`, `top_k`,
    `stop_sequences`, `candidate_count`, `seed`, `response_mime_type`, `tool_config`, `thinking_config`,
    `safety_settings`.
    """
    system_instruction: str | None = None  # system prompt / persona instructions
    top_p: float = 1.0  # nucleus sampling threshold
    top_k: int = 1  # restrict sampling to the top K candidate tokens
    stop_sequences: list[str] | None = None  # strings that halt generation early
    candidate_count: int = 1  # number of response candidates to generate
    seed: int | None = None  # deterministic sampling seed
    response_mime_type: str | None = None  # force structured output, e.g. "application/json"
    tool_config: dict[str, Any] | None = None  # controls how/whether tools are invoked
    thinking_config: dict[str, Any] | None = None  # extended thinking / reasoning budget config
    safety_settings: list[dict[str, Any]] | None = None  # per-category content-safety thresholds
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'system_instruction',
        'top_p',
        'top_k',
        'stop_sequences',
        'candidate_count',
        'seed',
        'response_mime_type',
        'tool_config',
        'thinking_config',
        'safety_settings'
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = genai.Client(api_key=self.api_key)

    def _call(self, prompt: str, **kwargs: Any) -> str:
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "tools": kwargs.get("tools", self.tools),
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                config_kwargs[args] = kwargs[args]

        response = self.model_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        logger.info(f"Google API response: {response}")
        return str(response.text or "").strip()


class GrokLLM(BaseLLM):
    """xAI Grok API caller, via the `openai` SDK pointed at `https://api.x.ai/v1` (OpenAI-compatible).

    No provider-specific extras wired up yet (only the base `tools` param) — candidates would include
    `top_p`, `stop`, `tool_choice`, `reasoning_effort`, and xAI's live-search `search_parameters`.
    """

    def _initialize_model(self) -> None:
        self.model_client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")

    def _call(self, prompt: str, **kwargs: Any) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.tools or kwargs.get("tools"):
            call_kwargs["tools"] = kwargs.get("tools", self.tools)

        response = self.model_client.chat.completions.create(**call_kwargs)
        logger.info(f"Grok API response: {response}")
        return str(response.choices[0].message.content or "").strip()


class OllamaLLM(BaseLLM):
    """Local Ollama server caller, via the `openai` SDK pointed at `http://localhost:11434/v1` (OpenAI-compatible).

    Extra params: `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `stream`, `tool_choice`, `seed`.
    """
    top_p: float = 1.0  # nucleus sampling threshold
    frequency_penalty: float = 0.0  # penalize tokens by how often they've already appeared
    presence_penalty: float = 0.0  # penalize tokens that have appeared at all so far
    stop: list[str] | None = None  # strings that halt generation early
    stream: bool = False  # stream the response instead of waiting for the full completion
    tool_choice: str | dict[str, Any] | None = None  # controls how/whether tools are invoked
    seed: int | None = None  # deterministic sampling seed
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'top_p', 
        'frequency_penalty', 
        'presence_penalty', 
        'stop', 
        'stream', 
        'tool_choice', 
        'seed'
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = OpenAI(api_key=self.api_key or "ollama", base_url="http://localhost:11434/v1")

    def _call(self, prompt: str, **kwargs: Any) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "messages": [{"role": "user", "content": prompt}],
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs.get(args, getattr(self, args, None))

        response = self.model_client.chat.completions.create(**call_kwargs)
        logger.info(f"Ollama response: {response}")
        return str(response.choices[0].message.content or "").strip()


class MistralLLM(BaseLLM):
    """Mistral AI API caller, via the `mistralai` SDK's `chat.complete`.

    Extra params: `top_p`, `random_seed`, `stop`, `tool_choice`, `presence_penalty`, `frequency_penalty`,
    `safe_prompt` (Mistral's built-in content moderation), `response_format`, `n`.
    """
    top_p: float = 1.0  # nucleus sampling threshold
    random_seed: int | None = None  # deterministic sampling seed
    stop: list[str] | None = None  # strings that halt generation early
    tool_choice: str | dict[str, Any] | None = None  # controls how/whether tools are invoked
    presence_penalty: float = 0.0  # penalize tokens that have appeared at all so far
    frequency_penalty: float = 0.0  # penalize tokens by how often they've already appeared
    safe_prompt: bool = False  # enable Mistral's built-in content moderation
    response_format: dict[str, Any] | None = None  # force structured output, e.g. JSON mode
    n: int | None = None  # number of completions to generate
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'top_p', 
        'random_seed', 
        'stop', 
        'tool_choice', 
        'presence_penalty', 
        'frequency_penalty', 
        'safe_prompt', 
        'response_format', 
        'n'
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = Mistral(api_key=self.api_key)

    def _call(self, prompt: str, **kwargs: Any) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "temperature": float(kwargs.get("temperature", self.temperature)),
            "messages": [{"role": "user", "content": prompt}],
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs.get(args, getattr(self, args, None)) 



        response = self.model_client.chat.complete(**call_kwargs)
        logger.info(f"Mistral API response: {response}")
        return str(response.choices[0].message.content or "").strip()
