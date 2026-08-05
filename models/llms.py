
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
from uilts.configs import LLMConfigs, resolve_secret


MAX_TOOL_RESULT_CHARS = 6000  # a tool result longer than this is truncated before it goes back to the model


def first_value(source: Any, *names: str, default: Any = '?') -> Any:
    """Read the first of `names` that `source` actually carries — one field under several providers' names."""
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value

    return default


@dataclass
class ToolCall:
    """One tool the model asked for, in the same shape whatever provider asked for it."""

    id: str  # the provider's id for this call, needed to match the result back to it
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolFailure:
    """A tool that raised. Carried as its own type so a provider can mark the result as an error.

    The failure goes back to the model rather than ending the run: a model that called a tool with a bad
    argument can read what went wrong and call it again, which is the whole point of returning it.
    """

    message: str


def as_tool_content(result: Any) -> str:
    """Render a tool's return value as the text the model reads back.

    Results are JSON where they can be, because that is what the tools return and what a model reads
    most reliably. A long one is truncated rather than dropped — a metric buried at the end of a 50KB
    result is worth less than the head of it plus an honest note that there was more.
    """
    if isinstance(result, ToolFailure):
        return f"Error: {result.message}"

    try:
        rendered = json.dumps(result, indent=2, default=str)
    except (TypeError, ValueError):
        rendered = str(result)

    if len(rendered) <= MAX_TOOL_RESULT_CHARS:
        return rendered

    return (
        f"{rendered[:MAX_TOOL_RESULT_CHARS]}\n\n"
        f"... [truncated: {len(rendered) - MAX_TOOL_RESULT_CHARS} more characters. "
        "Call the tool again with narrower arguments if you need the rest.]"
    )


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
        temperature: Sampling temperature, or None to leave it out of the request entirely — which is
            what the newest Claude models require, since they reject the parameter (Opus 5, Opus 4.8/4.7,
            Fable 5) or accept only their own default (Sonnet 5). Overridable per-call via kwargs.
        max_tokens: Max tokens to generate; overridable per-call via kwargs.
        api_key: Provider API key used to construct the client.
        tools: Tool/function-calling schema list, in the calling provider's expected format.
        type: Role this LLM plays ("generator", "retriever", "tool caller", "tool generator").
        mcp_servers: Remote MCP servers to connect for this call (currently used by ClaudeLLM only).
        **kwargs: Provider-specific overrides (e.g. `top_p`, `system`, `stream`) forwarded to
            `_initialize_model` and applied to matching class attributes on the subclass.
    """
    model_client: InferenceClient | anthropic.Anthropic | OpenAI | Mistral | genai.Client = None
    arguments: list[str] = []  # names of the optional params a subclass forwards to its provider

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

    def api_key_checker(self) -> None:
        """Resolve `api_key` as either an environment variable name or a literal key value."""
        try:
            self.api_key = resolve_secret(self.api_key, self.model_name)
        except ValueError as error:
            logger.error(str(error))
            raise

    @abstractmethod
    def _initialize_model(self, **kwargs: Any) -> None:
        """Initialize provider-specific model/client state."""

    def optional_arguments(self, *extra: str, **kwargs: Any) -> dict[str, Any]:
        """Collect the optional params to send: a per-call `kwargs` value first, else the one set at construction.

        A param is only sent when it actually has a value. Anything still None is one this caller was
        never configured with, and providers reject a null where they expect a value — an empty `tools`
        list included — so it is left out of the request rather than sent empty.

        `extra` names params outside the subclass's `arguments` list that belong in the same treatment,
        which is how `tools` (every provider) and `mcp_servers` (Claude only) are passed.
        """
        arguments: dict[str, Any] = {}
        for name in (*extra, *self.arguments):
            value = kwargs.get(name, getattr(self, name, None))
            if value is None or (isinstance(value, (list, tuple, dict)) and not value):
                continue
            arguments[name] = value

        return arguments

    def _call(self, prompt: str, **kwargs: Any) -> str:
        """Run one generation call and return plain text."""

    def log_response(self, stop_reason: str | None, usage: Any = None, detail: str = '') -> None:
        """Log one line about a response, and warn when the answer was cut off by `max_tokens`.

        Logging the whole response object buries the run: a single Claude reply carries kilobytes of
        base64 thinking signature. What is worth keeping is why it stopped and what it cost.

        Hitting `max_tokens` matters more than it looks. These models think by default and thinking is
        billed against the same ceiling, so a limit sized for the answer alone gets spent on reasoning
        and the answer arrives truncated — with no error, just a sentence that stops mid-word.
        """
        tokens = ''
        if usage is not None:
            # The same two numbers, named differently by each provider: Anthropic sends
            # `input_tokens`/`output_tokens`, OpenAI `prompt_tokens`/`completion_tokens`, Gemini
            # `prompt_token_count`/`candidates_token_count`. Reading only one set logs `?/?` for
            # everyone else, which is how a run ends up with no record of what it cost.
            sent = first_value(usage, "input_tokens", "prompt_tokens", "prompt_token_count")
            produced = first_value(usage, "output_tokens", "completion_tokens", "candidates_token_count")
            # Thinking tokens are nested one level down and named differently again: Anthropic under
            # `output_tokens_details.thinking_tokens`, OpenAI under
            # `completion_tokens_details.reasoning_tokens`, Gemini straight on the usage object.
            details = (
                getattr(usage, "output_tokens_details", None)
                or getattr(usage, "completion_tokens_details", None)
            )
            thinking = first_value(details, "thinking_tokens", "reasoning_tokens", default=None)
            thinking = thinking or getattr(usage, "thoughts_token_count", None)
            tokens = (
                f", tokens in/out={sent}/{produced}"
                f"{f' (thinking {thinking})' if thinking else ''}"
            )

        logger.info(f"{self.model_name}: stop_reason={stop_reason}{tokens}{detail}")

        # Providers spell the truncation reason differently too — `max_tokens`, `length`, `MAX_TOKENS`.
        if str(stop_reason).lower().removeprefix("finishreason.") in ("max_tokens", "length"):
            logger.warning(
                f"{self.model_name} hit max_tokens ({self.max_tokens}) and its answer is truncated. "
                "Thinking tokens count against the same ceiling — raise `max_tokens` for this llm."
            )

    # ---- tool-use round trip -----------------------------------------------------------------------
    # `_call` is text in, text out, which cannot run a tool: the model asks for one and the request ends.
    # These four methods are what a tool loop needs, and they are all a provider has to translate — the
    # loop itself lives once, in `BaseAgent.run_with_tools`. A caller that leaves `runs_tools` False is
    # called the old way, with its tools offered but never executed.
    runs_tools: bool = False

    def converse(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        """Run one turn of a tool-using conversation and return the provider's own response object."""
        raise NotImplementedError(f"{type(self).__name__} cannot run a tool-use conversation.")

    def read_turn(self, response: Any) -> tuple[str, list[ToolCall]]:
        """Split a response into the text it produced and the tools it asked for."""
        raise NotImplementedError(f"{type(self).__name__} cannot read a tool-use response.")

    def assistant_turn(self, response: Any) -> Any:
        """The response as a message to append, so the next turn sees what was already said."""
        raise NotImplementedError(f"{type(self).__name__} cannot replay an assistant turn.")

    def tool_result_turns(self, results: list[tuple[ToolCall, Any]]) -> list[Any]:
        """The executed results as messages to append — one provider wants one message, another wants one each."""
        raise NotImplementedError(f"{type(self).__name__} cannot return tool results.")


class OpenAIToolDialect:
    """The tool-use message shapes every OpenAI-compatible caller shares.

    OpenAI, Grok, Ollama, Mistral and the Hugging Face Inference API all speak the same dialect here: a
    response carries `choices[0].message.tool_calls`, each call's arguments arrive as a JSON *string*,
    and each result goes back as its own `{"role": "tool"}` message keyed by `tool_call_id`. Only the
    method used to send a request differs between them, and that stays in each caller's `converse`.
    """

    runs_tools = True

    def read_turn(self, response: Any) -> tuple[str, list[ToolCall]]:
        message = response.choices[0].message
        calls = []
        for call in (getattr(message, "tool_calls", None) or []):
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (TypeError, ValueError):  # a malformed argument string is the model's error to fix
                logger.warning(f"{call.function.name} was called with unparseable arguments; passing none.")
                arguments = {}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))

        return str(message.content or "").strip(), calls

    def assistant_turn(self, response: Any) -> dict[str, Any]:
        message = response.choices[0].message
        turn: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if getattr(message, "tool_calls", None):
            turn["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ]
        return turn

    def tool_result_turns(self, results: list[tuple[ToolCall, Any]]) -> list[dict[str, Any]]:
        return [
            {"role": "tool", "tool_call_id": call.id, "content": as_tool_content(result)}
            for call, result in results
        ]


class HuggingFaceInferenceLLM(OpenAIToolDialect, BaseLLM):
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
            "messages": [{"role": "user", "content": str(prompt)}],
            **self.optional_arguments('temperature', 'tools', **kwargs),
        }

        response = self.model_client.chat_completion(**call_kwargs)
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return str(response.choices[0].message.content or "").strip()

    def converse(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        response = self.model_client.chat_completion(
            model=self.model_name,
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            messages=messages,
            **self.optional_arguments('temperature', 'tools', tools=tools, **kwargs),
        )
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return response



class ClaudeLLM(BaseLLM):
    """Anthropic Claude API caller, via the `anthropic` SDK's `messages.create`.

    Extra params: `system`, `top_p`, `top_k`, `stop_sequences`, `stream`, `tool_choice`, `metadata`,
    `thinking`, `service_tier`, `container`. Also the only caller that forwards `mcp_servers`.
    """
    system: str | None = None  # system prompt / persona instructions
    top_p: float | None = None  # nucleus sampling threshold; Anthropic advises tuning this or temperature, not both
    top_k: int | None = None  # restrict sampling to the top K candidate tokens; leave unset to sample normally
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
            "messages": [{"role": "user", "content": prompt}],
            **self.optional_arguments('temperature', 'tools', 'mcp_servers', **kwargs),
        }

        response = self.model_client.messages.create(**call_kwargs)
        self.log_response(getattr(response, "stop_reason", None), getattr(response, "usage", None))

        # This caller returns text, so a response that stopped to call a tool has had its request
        # dropped: nothing here runs the tool and feeds the result back. Say so rather than returning
        # the half-finished text as though it were the whole answer.
        if getattr(response, "stop_reason", None) == "tool_use":
            asked_for = [block.name for block in response.content if block.type == "tool_use"]
            logger.warning(
                f"{self.model_name} stopped to call {asked_for}, which this caller does not run — "
                "the text below is only what it said before asking."
            )

        return "".join(block.text for block in response.content if block.type == "text").strip()

    # ---- tool-use round trip, in the Anthropic dialect ----------------------------------------------
    runs_tools = True

    def converse(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        response = self.model_client.messages.create(
            model=self.model_name,
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            messages=messages,
            **self.optional_arguments('temperature', 'tools', 'mcp_servers', tools=tools, **kwargs),
        )
        blocks = [block.type for block in response.content]
        self.log_response(
            getattr(response, "stop_reason", None), getattr(response, "usage", None), f", blocks={blocks}"
        )
        return response

    def read_turn(self, response: Any) -> tuple[str, list[ToolCall]]:
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
            for block in response.content
            if block.type == "tool_use"
        ]
        return text, calls

    def assistant_turn(self, response: Any) -> dict[str, Any]:
        # The blocks go back exactly as they arrived — a `tool_use` block must reach the next request
        # unchanged or its `tool_result` has nothing to attach to.
        return {"role": "assistant", "content": response.content}

    def tool_result_turns(self, results: list[tuple[ToolCall, Any]]) -> list[dict[str, Any]]:
        # Anthropic wants every result for a turn in one user message, not one message each.
        return [{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": as_tool_content(result),
                    **({"is_error": True} if isinstance(result, ToolFailure) else {}),
                }
                for call, result in results
            ],
        }]


class OpenAILLM(OpenAIToolDialect, BaseLLM):
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
            "messages": [{"role": "user", "content": prompt}],
            **self.optional_arguments('temperature', 'tools', **kwargs),
        }

        response = self.model_client.chat.completions.create(**call_kwargs)
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )

        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":  # see the note in ClaudeLLM._call
            asked_for = [call.function.name for call in (choice.message.tool_calls or [])]
            logger.warning(
                f"{self.model_name} stopped to call {asked_for}, which this caller does not run — "
                "the text below is only what it said before asking."
            )

        return str(choice.message.content or "").strip()

    def converse(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        response = self.model_client.chat.completions.create(
            model=self.model_name,
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            messages=messages,
            **self.optional_arguments('temperature', 'tools', tools=tools, **kwargs),
        )
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return response



class GoogleLLM(BaseLLM):
    """Google Gemini API caller, via the `google-genai` SDK's `models.generate_content`.

    Extra params (passed through `GenerateContentConfig`): `system_instruction`, `top_p`, `top_k`,
    `stop_sequences`, `candidate_count`, `seed`, `response_mime_type`, `tool_config`, `thinking_config`,
    `safety_settings`.
    """
    system_instruction: str | None = None  # system prompt / persona instructions
    top_p: float | None = None  # nucleus sampling threshold
    top_k: int | None = None  # restrict sampling to the top K candidate tokens; leave unset to sample normally
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
            **self.optional_arguments('temperature', 'tools', **kwargs),
        }

        response = self.model_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        self.log_response(self.finish_reason(response), getattr(response, "usage_metadata", None))

        # Read the parts rather than `response.text`: a response mixing text with a function call has no
        # `.text` at all, so a turn that asked for a tool used to come back as zero characters of output.
        asked_for = [
            part.function_call.name for part in self.parts_of(response)
            if getattr(part, "function_call", None)
        ]
        if asked_for:
            logger.warning(
                f"{self.model_name} stopped to call {asked_for}, which this caller does not run — "
                "the text below is only what it said before asking."
            )

        return self.text_of(response)

    # ---- tool-use round trip, in the Gemini dialect --------------------------------------------------
    runs_tools = True

    @staticmethod
    def parts_of(response: Any) -> list[Any]:
        """The parts of the first candidate — where Gemini puts both the text and the function calls."""
        candidates = getattr(response, "candidates", None) or []
        content = getattr(candidates[0], "content", None) if candidates else None
        return list(getattr(content, "parts", None) or [])

    @classmethod
    def text_of(cls, response: Any) -> str:
        """Every text part joined — what the model actually said, whatever else the turn carried."""
        return "".join(
            part.text for part in cls.parts_of(response) if getattr(part, "text", None)
        ).strip()

    @staticmethod
    def finish_reason(response: Any) -> str | None:
        """Why generation stopped. Gemini reports it per candidate, not on the response itself."""
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        if reason is None:
            return None

        return str(getattr(reason, "name", None) or reason)  # "MAX_TOKENS", not "FinishReason.MAX_TOKENS"

    def as_contents(self, messages: list[Any]) -> list[Any]:
        """Translate the loop's messages into the `contents` Gemini takes.

        The loop opens with `{"role": "user", "content": prompt}` — the shape Anthropic and OpenAI both
        accept — and Gemini rejects it outright: it wants `{"role": "user", "parts": [{"text": ...}]}`.
        The turns this class appends are already in that shape and pass through untouched, so this only
        rewrites the opening message, and renames `assistant` to the `model` role Gemini uses for its own.
        """
        contents = []
        for message in messages:
            if not isinstance(message, dict) or "parts" in message:
                contents.append(message)  # a Content object, or a turn this class already built
                continue

            role = "model" if message.get("role") == "assistant" else message.get("role", "user")
            contents.append({"role": role, "parts": [{"text": str(message.get("content", ''))}]})

        return contents

    def converse(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": int(kwargs.pop("max_tokens", self.max_tokens)),
            **self.optional_arguments('temperature', 'tools', tools=tools, **kwargs),
        }

        response = self.model_client.models.generate_content(
            model=self.model_name,
            contents=self.as_contents(messages),
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        parts = [
            "function_call" if getattr(part, "function_call", None) else "text"
            for part in self.parts_of(response)
        ]
        self.log_response(
            self.finish_reason(response), getattr(response, "usage_metadata", None), f", parts={parts}"
        )
        return response

    def read_turn(self, response: Any) -> tuple[str, list[ToolCall]]:
        calls = []
        for position, part in enumerate(self.parts_of(response)):
            call = getattr(part, "function_call", None)
            if not call:
                continue

            # Gemini matches a result back to its call by function name, and only some surfaces send an
            # id at all — so one is made up when it is absent, since the loop keys its record on it.
            calls.append(ToolCall(
                id=getattr(call, "id", None) or f"{call.name}-{position}",
                name=call.name,
                arguments=dict(call.args or {}),
            ))

        return self.text_of(response), calls

    def assistant_turn(self, response: Any) -> Any:
        """Return the response as a message to append, so the next turn sees what was already said.
        """
        # The candidate's own content goes back exactly as it arrived — a `function_call` part has to
        # reach the next request unchanged or the `function_response` answering it has nothing to attach
        # to, and Gemini rejects the turn.
        candidates = getattr(response, "candidates", None) or []
        content = getattr(candidates[0], "content", None) if candidates else None
        return content or {"role": "model", "parts": [{"text": self.text_of(response)}]}

    def tool_result_turns(self, results: list[tuple[ToolCall, Any]]) -> list[dict[str, Any]]:
        # Gemini takes every result for a turn in one user message, each keyed by the function's name
        # rather than by a call id, and each response is a mapping rather than a string.
        return [{
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": call.name,
                        "response": (
                            {"error": result.message}
                            if isinstance(result, ToolFailure)
                            else {"result": as_tool_content(result)}
                        ),
                    }
                }
                for call, result in results
            ],
        }]


class GrokLLM(OpenAIToolDialect, BaseLLM):
    """xAI Grok API caller, via the `openai` SDK pointed at `https://api.x.ai/v1` (OpenAI-compatible).

    No provider-specific extras wired up yet (only the base `tools` param) — candidates would include
    `top_p`, `stop`, `tool_choice`, `reasoning_effort`, and xAI's live-search `search_parameters`.
    """

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = OpenAI(api_key=self.api_key, base_url="https://api.x.ai/v1")

    def _call(self, prompt: str, **kwargs: Any) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(kwargs.get("max_tokens", self.max_tokens)),
            "messages": [{"role": "user", "content": prompt}],
            **self.optional_arguments('temperature', 'tools', **kwargs),
        }

        response = self.model_client.chat.completions.create(**call_kwargs)
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return str(response.choices[0].message.content or "").strip()

    def converse(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        """Run one turn of a tool-using conversation and return the provider's own response object."""
        response = self.model_client.chat.completions.create(
            model=self.model_name,
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            messages=messages,
            **self.optional_arguments('temperature', 'tools', tools=tools, **kwargs),
        )
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return response


class OllamaLLM(OpenAIToolDialect, BaseLLM):
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
            "messages": [{"role": "user", "content": prompt}],
            **self.optional_arguments('temperature', 'tools', **kwargs),
        }

        response = self.model_client.chat.completions.create(**call_kwargs)
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return str(response.choices[0].message.content or "").strip()

    def converse(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        response = self.model_client.chat.completions.create(
            model=self.model_name,
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            messages=messages,
            **self.optional_arguments('temperature', 'tools', tools=tools, **kwargs),
        )
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return response



class MistralLLM(OpenAIToolDialect, BaseLLM):
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
            "messages": [{"role": "user", "content": prompt}],
            **self.optional_arguments('temperature', 'tools', **kwargs),
        }

        response = self.model_client.chat.complete(**call_kwargs)
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return str(response.choices[0].message.content or "").strip()

    def converse(self, messages: list[Any], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        response = self.model_client.chat.complete(
            model=self.model_name,
            max_tokens=int(kwargs.pop("max_tokens", self.max_tokens)),
            messages=messages,
            **self.optional_arguments('temperature', 'tools', tools=tools, **kwargs),
        )
        self.log_response(
            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None)
        )
        return response
