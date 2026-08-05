
import hashlib
import math
import re
from abc import abstractmethod
from collections import Counter
from typing import Any

from openai import OpenAI
from mistralai.client import Mistral
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types as genai_types

from uilts.configs import EmbeddingsConfigs, resolve_secret
from uilts.logger import logger


class BaseEmbeddings(EmbeddingsConfigs):
    """Common interface for provider-specific embeddings callers (OpenAI, Google, HuggingFace, Mistral, Ollama, ...).

    Subclasses implement two methods:
      - `_initialize_model(**kwargs)`: build the provider's SDK client and assign it to `self.model_client`.
        Any kwarg matching an existing class attribute (e.g. `dimensions`, `task_type`) is applied via
        `setattr`, so provider-specific options can be set at construction time.
      - `_call(text, **kwargs)`: embed a single text against that client and return the embedding vector.

    Provider-specific optional parameters are declared as class attributes with defaults, and their names
    are listed in a class-level `arguments` list. `_call` walks that list and forwards each one into the
    API request, preferring a per-call `kwargs` value over the instance default set at construction. This
    lets every param be configured once and still be overridden per-call without changing the method
    signature.

    Args:
        model_name: Provider/model identifier (e.g. "text-embedding-3-small", "gemini-embedding-001").
        api_key: Provider API key used to construct the client.
        **kwargs: Provider-specific overrides forwarded to `_initialize_model` and applied to matching
            class attributes on the subclass.
    """
    model_client: Any = None

    def __init__(self, model_name: str, api_key: str, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, api_key=api_key)
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

    def _call(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single text and return its embedding vector."""

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        """Embed one text — what a caller asks for, over the `_call` each provider implements."""
        return self._call(text, **kwargs)

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Embed several texts, one call each — how a knowledge base is written in the first place."""
        return [self._call(text, **kwargs) for text in texts]


class LocalEmbeddings(BaseEmbeddings):
    """Embeddings computed in-process — hashed bag of words. No provider, no key, no download.

    Every other caller here is a client for someone's API. This one is the vector itself: the text is
    tokenised into words and word pairs, each is hashed to a coordinate and a sign, weighted by log term
    frequency, and the result is L2-normalised. The cosine similarity between two of these is a weighted
    term-overlap score — the hashing trick, which is how a TF-IDF retriever works at a fixed width with
    no vocabulary to fit.

    It is not a trained model and does not pretend to be one: it matches documents that share wording,
    not documents that share meaning, so a paraphrase with nothing in common lexically scores zero. Over
    a small curated knowledge base whose questions use the vocabulary the notes use, that is usually
    enough — and it is what makes a retrieval pipeline runnable offline, in CI, and before any key
    exists. Move to a trained model (`openai/text-embedding-3-small`, `google/gemini-embedding-001`,
    a local `ollama/...`) when meaning has to be matched rather than words.

    Hashing is `blake2b` rather than the built-in `hash`, which is salted per process: the same text
    must give the same vector on the next run, or an index written today cannot be searched tomorrow.

    The width comes from the model id — `local/hashing-1024` is 1024-wide — or from `dimensions`,
    defaulting to 512. It has to match the `dimension` of the vector db these are written into.
    """

    dimensions: int | None = None  # width of the vector; also settable as a `-<width>` suffix on the id
    arguments: list[str] = [
        'dimensions',
    ]

    WORDS = re.compile(r"[a-z0-9]+")
    DEFAULT_WIDTH = 512

    def api_key_checker(self) -> None:
        """No credential to resolve: nothing leaves the process, so a configured key is simply unused."""
        self.api_key = ''

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.dimensions = int(self.dimensions or self.width_from(self.model_name) or self.DEFAULT_WIDTH)
        self.model_client = None  # there is no client; `_call` below is the whole model
        logger.info(
            f"Local embeddings ready: {self.dimensions}-wide hashed bag of words, nothing is called."
        )

    @staticmethod
    def width_from(model_name: str) -> int | None:
        """Read the width off the model id, so `hashing-3072` is 3072-wide."""
        tail = str(model_name).rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else None

    def tokens(self, text: str) -> list[str]:
        """Words plus adjacent word pairs — the pairs are what keep 'not imbalanced' from matching 'imbalanced'."""
        words = self.WORDS.findall(text.lower())
        return words + [f"{first}_{second}" for first, second in zip(words, words[1:])]

    def _call(self, text: str, **kwargs: Any) -> list[float]:
        width = int(kwargs.get("dimensions") or self.dimensions)
        vector = [0.0] * width

        for token, count in Counter(self.tokens(text)).items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % width
            # A sign per token, so two different tokens landing on one coordinate cancel as often as
            # they add, instead of every collision inflating the similarity.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector] if length else vector


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI Embeddings API caller, via the `openai` SDK's `embeddings.create`.

    Extra params: `dimensions`, `encoding_format`, `user`.
    """
    dimensions: int | None = None  # truncate the output embedding to this many dimensions
    encoding_format: str | None = None  # "float" or "base64"
    user: str | None = None  # end-user identifier for abuse tracking
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'dimensions',
        'encoding_format',
        'user',
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = OpenAI(api_key=self.api_key)

    def _call(self, text: str, **kwargs: Any) -> list[float]:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": text,
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs.get(args, getattr(self, args, None))

        response = self.model_client.embeddings.create(**call_kwargs)
        logger.info(f"OpenAI embeddings response: {response}")
        return list(response.data[0].embedding)


class GoogleEmbeddings(BaseEmbeddings):
    """Google Gemini embeddings API caller, via the `google-genai` SDK's `models.embed_content`.

    Extra params (passed through `EmbedContentConfig`): `task_type`, `title`, `output_dimensionality`.
    """
    task_type: str | None = None  # e.g. "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY"
    title: str | None = None  # document title, only used with task_type="RETRIEVAL_DOCUMENT"
    output_dimensionality: int | None = None  # truncate the output embedding to this many dimensions
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'task_type',
        'title',
        'output_dimensionality',
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = genai.Client(api_key=self.api_key)

    def _call(self, text: str, **kwargs: Any) -> list[float]:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "contents": text,
        }
        config_kwargs: dict[str, Any] = {}
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                config_kwargs[args] = kwargs.get(args, getattr(self, args, None))
        if config_kwargs:
            call_kwargs["config"] = genai_types.EmbedContentConfig(**config_kwargs)

        response = self.model_client.models.embed_content(**call_kwargs)
        logger.info(f"Google embeddings response: {response}")
        return list(response.embeddings[0].values)


class HuggingFaceEmbeddings(BaseEmbeddings):
    """Hugging Face Inference API embeddings caller, via `huggingface_hub.InferenceClient.feature_extraction`.

    Extra params: `normalize`, `truncate`, `truncation_direction`.
    """
    normalize: bool | None = None  # L2-normalize the returned embedding
    truncate: bool | None = None  # truncate input text to the model's max sequence length
    truncation_direction: str | None = None  # "Left" or "Right", which side to truncate from
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'normalize',
        'truncate',
        'truncation_direction',
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = InferenceClient(api_key=self.api_key)

    def _call(self, text: str, **kwargs: Any) -> list[float]:
        call_kwargs: dict[str, Any] = {
            "text": text,
            "model": self.model_name,
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs.get(args, getattr(self, args, None))

        response = self.model_client.feature_extraction(**call_kwargs)
        logger.info(f"HF embeddings response: {response}")
        return list(response)


class MistralEmbeddings(BaseEmbeddings):
    """Mistral AI embeddings API caller, via the `mistralai` SDK's `embeddings.create`.

    Extra params: `output_dimension`, `output_dtype`.
    """
    output_dimension: int | None = None  # truncate the output embedding to this many dimensions
    output_dtype: str | None = None  # e.g. "float", "int8", "uint8", "binary", "ubinary"
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'output_dimension',
        'output_dtype',
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = Mistral(api_key=self.api_key)

    def _call(self, text: str, **kwargs: Any) -> list[float]:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "inputs": [text],
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs.get(args, getattr(self, args, None))

        response = self.model_client.embeddings.create(**call_kwargs)
        logger.info(f"Mistral embeddings response: {response}")
        return list(response.data[0].embedding)


class OllamaEmbeddings(BaseEmbeddings):
    """Local Ollama server embeddings caller, via the `openai` SDK pointed at `http://localhost:11434/v1` (OpenAI-compatible).

    Extra params: `dimensions`, `encoding_format`, `user`.
    """
    dimensions: int | None = None  # truncate the output embedding to this many dimensions
    encoding_format: str | None = None  # "float" or "base64"
    user: str | None = None  # end-user identifier for abuse tracking
    arguments: list[str] = [  # names of the optional params above, forwarded to the API call when set
        'dimensions',
        'encoding_format',
        'user',
    ]

    def _initialize_model(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.model_client = OpenAI(api_key=self.api_key or "ollama", base_url="http://localhost:11434/v1")

    def _call(self, text: str, **kwargs: Any) -> list[float]:
        call_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": text,
        }
        for args in self.arguments:
            if kwargs.get(args) or getattr(self, args, None) is not None:
                call_kwargs[args] = kwargs.get(args, getattr(self, args, None))

        response = self.model_client.embeddings.create(**call_kwargs)
        logger.info(f"Ollama embeddings response: {response}")
        return list(response.data[0].embedding)
