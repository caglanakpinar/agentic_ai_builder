"""The builders behind every piece of a pipeline: agents, provider callers, and db connectors.

This is the layer the `agentic-ai` command drives, exposed as plain functions so a pipeline can just as
well be assembled in Python:

    from agent_builder import build_agent, load_configs

    configs = load_configs("my_pipeline")
    agent = build_agent("triage_agent", configs)
    print(agent.run(question="the login page 500s", context="", agent_outputs={}))

Every builder takes its values from a configured entry — looked up by `name` in the `Configs` it is
given — from keyword overrides, or from both, and an override always wins over what the YAML says. Given
no config at all, the keywords have to describe the object on their own, which is how a provider or an
engine gets tried out before any config is written for it.

Anything a builder cannot build from is raised as `ValueError`, and a driver that isn't installed as
`ImportError`, so a caller can handle bad config without depending on the CLI.

Provider SDKs and db drivers are imported lazily, inside the builder that needs one: the registries below
hold dotted paths rather than classes, so importing this module pulls in no provider and no driver.
"""

import importlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from uilts.configs import (
    AgentConfigs,
    Configs,
    SQLDBConfigs,
    TextDBConfigs,
    VectorDBConfigs,
)
from uilts.logger import logger


# Registries: the name written in the YAML (or passed as `provider`/`db`) -> the class implementing it.
# Values are dotted paths rather than classes so nothing is imported until a builder resolves one.
LLM_CALLERS: dict[str, str] = {
    "claude": "models.llms.ClaudeLLM",
    "anthropic": "models.llms.ClaudeLLM",
    "openai": "models.llms.OpenAILLM",
    "google": "models.llms.GoogleLLM",
    "gemini": "models.llms.GoogleLLM",
    "grok": "models.llms.GrokLLM",
    "xai": "models.llms.GrokLLM",
    "ollama": "models.llms.OllamaLLM",
    "mistral": "models.llms.MistralLLM",
    "huggingface": "models.llms.HuggingFaceInferenceLLM",
    "hf": "models.llms.HuggingFaceInferenceLLM",
}

EMBEDDINGS: dict[str, str] = {
    "openai": "models.embeddings.OpenAIEmbeddings",
    "google": "models.embeddings.GoogleEmbeddings",
    "gemini": "models.embeddings.GoogleEmbeddings",
    "huggingface": "models.embeddings.HuggingFaceEmbeddings",
    "hf": "models.embeddings.HuggingFaceEmbeddings",
    "mistral": "models.embeddings.MistralEmbeddings",
    "ollama": "models.embeddings.OllamaEmbeddings",
}

VECTOR_DBS: dict[str, str] = {
    "faiss": "db_connector.vector.FAISSDB",
    "chroma": "db_connector.vector.ChromaDB",
    "qdrant": "db_connector.vector.QdrantDB",
    "pinecone": "db_connector.vector.PineconeDB",
    "weaviate": "db_connector.vector.WeaviateDB",
    "milvus": "db_connector.vector.MilvusDB",
    "lancedb": "db_connector.vector.LanceDB",
}

TEXT_DBS: dict[str, str] = {
    "elasticsearch": "db_connector.text.ElasticsearchTextDB",
    "opensearch": "db_connector.text.OpenSearchTextDB",
    "meilisearch": "db_connector.text.MeilisearchTextDB",
    "typesense": "db_connector.text.TypesenseTextDB",
}

SQL_DBS: dict[str, str] = {
    "postgresql": "db_connector.tabular.PostgreSQLDB",
    "postgres": "db_connector.tabular.PostgreSQLDB",
    "mysql": "db_connector.tabular.MySQLDB",
    "sqlite": "db_connector.tabular.SQLiteDB",
    "duckdb": "db_connector.tabular.DuckDBDB",
    "snowflake": "db_connector.tabular.SnowflakeDB",
    "redshift": "db_connector.tabular.RedshiftDB",
    "bigquery": "db_connector.tabular.BigQueryDB",
}

AGENT_TYPES: dict[str, str] = {  # `agents.<name>.type` in the YAML -> the class that runs that role
    "generator": "builder.agents.WorkerAgent",
    "worker": "builder.agents.WorkerAgent",
    "judger": "builder.agents.JudgerAgent",
    "classifier": "builder.agents.ClassifierAgent",
    "planner": "builder.agents.PlannerAgent",
    "thinker": "builder.agents.PlannerAgent",
    "rag": "builder.agents.RAGBuilderAgent",
    "rag_builder": "builder.agents.RAGBuilderAgent",
    "retriever": "builder.agents.RAGBuilderAgent",
}

# db category -> the engines it can be connected with, and the configs whose fields that connector takes.
DB_CONNECTORS: dict[str, tuple[dict[str, str], type]] = {
    "vector": (VECTOR_DBS, VectorDBConfigs),
    "text": (TEXT_DBS, TextDBConfigs),
    "sql": (SQL_DBS, SQLDBConfigs),
}

PROVIDER_HINTS: dict[str, str] = {  # substring of a model id -> the provider that serves it
    "claude": "claude",
    "anthropic": "claude",
    "gpt": "openai",
    "text-embedding": "openai",
    "gemini": "google",
    "bison": "google",
    "gecko": "google",
    "vertexai": "google",
    "grok": "grok",
    "mixtral": "mistral",
    "mistral": "mistral",
    "ollama": "ollama",
    "llama": "huggingface",
}

DEFAULT_MAX_TOKENS = 1024  # BaseLLM has no default of its own, and every provider requires the field


def load_class(path: str) -> type:
    """Import the class a dotted registry path names, e.g. `"models.llms.ClaudeLLM"`."""
    module_path, class_name = path.rsplit(".", 1)
    try:
        return getattr(importlib.import_module(module_path), class_name)
    except ImportError as error:  # the driver or SDK the class needs isn't installed
        raise ImportError(
            f"cannot import {path}: {error}. Install the extra that provides it, "
            f'e.g. `poetry install --extras "chroma"`.'
        ) from error


def resolve(registry: dict[str, str], key: str, label: str) -> type:
    """Look `key` up in one of the registries above and import the class it names."""
    if str(key).lower() not in registry:
        raise ValueError(f"unknown {label} {key!r}; expected one of {sorted(registry)}.")

    return load_class(registry[str(key).lower()])


def split_model(model_name: str, registry: dict[str, str]) -> tuple[str | None, str]:
    """Split a `provider/model` id: `"google/gemini-2.5-pro"` -> `("google", "gemini-2.5-pro")`.

    Only the leading segment is read as a provider, and only when it names one, so a Hugging Face id
    like `"mistralai/Mistral-7B-Instruct"` — whose first segment is an org, not a provider — is left
    alone and passed to the SDK whole.
    """
    head, separator, tail = model_name.partition("/")
    if separator and head.lower() in registry:
        return head.lower(), tail

    return None, model_name


def resolve_provider(
    provider: str | None,
    model_name: str,
    registry: dict[str, str],
    label: str,
) -> tuple[str, str]:
    """Return the `(provider, model id)` to build with, inferring the provider when it isn't given.

    The provider comes from `provider` when passed, otherwise from a `provider/` prefix on the model id,
    otherwise from a substring of the id itself (`PROVIDER_HINTS`) — which is what makes
    `model_name="claude-sonnet-5"` enough on its own.
    """
    prefix, model_id = split_model(model_name, registry)
    if provider:
        return provider.lower(), model_id

    if prefix:
        return prefix, model_id

    for hint, name in PROVIDER_HINTS.items():
        if hint in model_name.lower():
            return name, model_id

    raise ValueError(f"cannot tell which {label} serves {model_name!r}; name one with `provider`.")


def given(**overrides: Any) -> dict[str, Any]:
    """Keep only the overrides that were actually passed, so an unset one can't blank a configured value."""
    return {key: value for key, value in overrides.items() if value not in (None, (), [])}


def load_configs(config_dir: str | Path) -> Configs:
    """Read the YAML config directory into the `Configs` every builder takes its defaults from."""
    return Configs(str(config_dir))


def as_configs(configs: Configs | str | Path | None) -> tuple[Configs | None, str]:
    """Accept either a config directory or an already-read `Configs`, and return both.

    The directory travels alongside the config because an agent needs it too: its prompts and its tools
    are resolved relative to wherever the YAML lives.
    """
    if isinstance(configs, Configs):
        return configs, str(configs.current_dir)

    if configs:
        return load_configs(configs), str(configs)

    return None, "."


def configured(registry: dict[str, Any] | None, name: str | None, label: str) -> dict[str, Any]:
    """Return a configured entry as the dict of defaults the overrides are then applied over.

    A `name` that no config defines is not an error: it names something being built from keywords alone,
    which is only reported as missing later, if a required field turns out to have nowhere to come from.
    """
    if not name or not registry or name not in registry:
        if name and registry:
            logger.info(f"No {label} named {name!r} in the config; building it from arguments alone.")
        return {}

    return asdict(registry[name])


def optional(build: Any, label: str, **kwargs: Any) -> Any:
    """Run one of the builders below, downgrading a failure to a warning.

    An agent's retrieval dependencies are optional in exactly this sense: `RAGBuilderAgent` already falls
    back to the caller-supplied context when a connector is missing, so a db that isn't reachable yet
    should leave the agent buildable rather than fail the whole build.
    """
    try:
        return build(**kwargs)
    except Exception as error:
        logger.warning(f"Could not build {label}: {error}")
        return None


def build_llm(
    name: str | None = None,
    configs: Configs | str | Path | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    type: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    mcp_servers: Sequence[str] = (),
    settings: dict[str, Any] | None = None,
) -> Any:
    """Build one provider caller from a configured `llms:` entry, from arguments, or from both.

    Args:
        name: Key the caller is listed under in the `llms:` block.
        configs: Config directory or an already-read `Configs` to look `name` up in.
        provider: Who serves the model. Inferred from the model id when not given.
        model_name: Model id to call, optionally prefixed with `provider/`.
        api_key: The key itself, or the name of the environment variable holding it.
        temperature, max_tokens, type, tools, mcp_servers: The rest of `LLMConfigs`.
        settings: Provider-specific options (`top_p`, `thinking`, …) passed to the caller's constructor.

    Returns:
        A `BaseLLM` subclass instance with its SDK client already constructed.
    """
    configs, _ = as_configs(configs)
    fields = configured(configs.llm_configs if configs else None, name, "llm")
    fields.update(given(
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        type=type,
        tools=tools,
        mcp_servers=list(mcp_servers),
    ))

    if not fields.get("model_name"):
        raise ValueError(
            f"no model to call for {name or 'the llm'!r}: pass `model_name`, "
            "or a `name` listed in the `llms:` config."
        )

    provider, model_id = resolve_provider(provider, fields["model_name"], LLM_CALLERS, "provider")
    caller = resolve(LLM_CALLERS, provider, "provider")
    return caller(
        model_name=model_id,
        # Only sent when the config names one — the newest Claude models reject `temperature`.
        temperature=None if fields.get("temperature") is None else float(fields["temperature"]),
        max_tokens=int(fields.get("max_tokens") or DEFAULT_MAX_TOKENS),
        api_key=fields.get("api_key") or "",
        tools=fields.get("tools"),
        type=fields.get("type") or "generator",
        mcp_servers=fields.get("mcp_servers") or None,
        **(settings or {}),
    )


def build_embeddings(
    name: str | None = None,
    configs: Configs | str | Path | None = None,
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    settings: dict[str, Any] | None = None,
) -> Any:
    """Build one embeddings caller — what a RAG agent turns its question into a vector with.

    Takes the same shape of arguments as `build_llm`, against the `embeddings:` block.
    """
    configs, _ = as_configs(configs)
    fields = configured(configs.embeddings_configs if configs else None, name, "embeddings")
    fields.update(given(model_name=model_name, api_key=api_key))

    if not fields.get("model_name"):
        raise ValueError(
            f"no model to embed with for {name or 'the embeddings'!r}: pass `model_name`, "
            "or a `name` listed in the `embeddings:` config."
        )

    provider, model_id = resolve_provider(provider, fields["model_name"], EMBEDDINGS, "embeddings provider")
    caller = resolve(EMBEDDINGS, provider, "embeddings provider")
    return caller(model_name=model_id, api_key=fields.get("api_key") or "", **(settings or {}))


def build_db(
    name: str | None = None,
    configs: Configs | str | Path | None = None,
    category: str = "vector",
    settings: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Build and connect one db connector from a configured `dbs:` entry, from arguments, or from both.

    Args:
        name: Key the db is listed under in the `dbs:` block.
        configs: Config directory or an already-read `Configs` to look `name` up in.
        category: Which kind of db to build — "vector", "text" or "sql". A configured entry declaring a
            different `type` is refused, rather than connected as something it isn't.
        settings: Driver-specific options (`nlist`, `prefer_grpc`, …) passed to the connector.
        **overrides: Any field of the category's configs — `db`, `host`, `path`, `dimension`, … — each
            overriding what the configured entry says.

    Returns:
        A connected connector: `BaseVectorDB`, `BaseTextDB` or `BaseSQLDB`, by category.
    """
    if category not in DB_CONNECTORS:
        raise ValueError(f"unknown db category {category!r}; expected one of {sorted(DB_CONNECTORS)}.")

    registry, declared = DB_CONNECTORS[category]
    accepted = set(declared.__dataclass_fields__) - {"type"}  # `type` is the category, passed separately
    unknown = set(overrides) - accepted
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} are not {category} db settings; pass driver-specific options as "
            f"`settings`. A {category} db takes {sorted(accepted)}."
        )

    configs, _ = as_configs(configs)
    fields = configured(configs.db_confgs if configs else None, name, "db")
    if fields and fields.get("type") != category:
        raise ValueError(f"db {name!r} is a {fields.get('type')!r} db, not a {category} one.")

    fields.update(given(name=name, **overrides))
    if not fields.get("db"):
        raise ValueError(
            f"no engine to connect with for {name or category!r}: pass `db`, "
            "or a `name` listed in the `dbs:` config."
        )

    connector = resolve(registry, fields["db"], f"{category} db engine")
    fields["name"] = fields.get("name") or f"{category}_db"
    fields["db"] = str(fields["db"]).lower()
    return connector(
        # `type` is the category, which the connector class already is — every other configured field is
        # a constructor argument.
        **{key: value for key, value in fields.items() if key != "type"},
        **(settings or {}),
    )


def build_vector_db(
    name: str | None = None,
    configs: Configs | str | Path | None = None,
    settings: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Build one vector db connector (FAISS, Chroma, Qdrant, Pinecone, Weaviate, Milvus, LanceDB)."""
    return build_db(name, configs, category="vector", settings=settings, **overrides)


def build_text_db(
    name: str | None = None,
    configs: Configs | str | Path | None = None,
    settings: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Build one text db connector (Elasticsearch, OpenSearch, Meilisearch, Typesense)."""
    return build_db(name, configs, category="text", settings=settings, **overrides)


def build_sql_db(
    name: str | None = None,
    configs: Configs | str | Path | None = None,
    settings: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Build one SQL db connector (PostgreSQL, MySQL, SQLite, DuckDB, Snowflake, Redshift, BigQuery)."""
    return build_db(name, configs, category="sql", settings=settings, **overrides)


def build_agent(
    name: str,
    configs: Configs | str | Path | None = None,
    llm: Any = None,
    substitute_llm: Any = None,
    embeddings: str | None = None,
    labels: Sequence[str] = (),
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    settings: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Build one agent from a configured `agents:` entry, from arguments, or from both.

    The agent's `type` picks the class that runs it (`AGENT_TYPES`), its tools are imported and its
    prompt directory is located while it is built, and a RAG agent's vector db, text db and embeddings
    caller are wired up — one that can't be built is warned about and left unwired rather than failing
    the build, since `RAGBuilderAgent` falls back to the context it is called with.

    Args:
        name: Agent name, and the key it is looked up under in the `agents:` block.
        configs: Config directory or an already-read `Configs`. Doubles as the root the agent's prompt
            and tool paths resolve against, so an agent with either needs one.
        llm: The caller backing this agent — a name from `llms:`, or an already-built `BaseLLM` to use
            as it is. Defaults to whatever the configured entry names.
        substitute_llm: Caller to fall back to when the primary one raises, in the same two forms.
        embeddings: Name from `embeddings:` a RAG agent embeds its question with.
        labels: The labels a classifier agent is allowed to answer with.
        provider, model_name, api_key, temperature, max_tokens, settings: Passed to `build_llm` when the
            caller is built here rather than handed in, and override the configured `llms:` entry.
        **overrides: Any field of `AgentConfigs` — `type`, `prompt`, `tools`, `mcp_servers`, `db_vector`,
            `db_text`, `db_sql` — each overriding what the configured entry says.

    Returns:
        A `BaseAgent` subclass instance, ready for `run(question, context, agent_outputs)`.
    """
    unknown = set(overrides) - set(AgentConfigs.__dataclass_fields__)
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} are not agent settings. An agent takes "
            f"{sorted(AgentConfigs.__dataclass_fields__)}."
        )

    configs, config_dir = as_configs(configs)
    fields = configured(configs.agent_configs if configs else None, name, "agent")
    entry = dict((getattr(configs, "agents", None) or {}).get(name) or {}) if configs else {}
    fields.update(given(**{
        key: list(value) if isinstance(value, (tuple, set)) else value
        for key, value in overrides.items()
    }))

    agent_class = resolve(AGENT_TYPES, fields.get("type") or "generator", "agent type")
    fields.setdefault("responsiblity_prompt", '')  # the one AgentConfigs field with no default of its own
    agent_config = AgentConfigs(**{
        key: value for key, value in fields.items() if key in AgentConfigs.__dataclass_fields__
    })

    if not hasattr(llm, "_call"):  # a name, or nothing — so the caller is built here
        llm = build_llm(
            name=llm or entry.get("llm"),
            configs=configs,
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            mcp_servers=agent_config.mcp_servers or (),
            settings=settings,
        )

    fallback = substitute_llm or entry.get("substitute_llm")
    if fallback and not hasattr(fallback, "_call"):
        fallback = optional(build_llm, f"substitute llm {fallback!r}", name=fallback, configs=configs)

    kwargs: dict[str, Any] = {}
    if agent_class.__name__ == "RAGBuilderAgent":  # retrieval is this type's whole job, so wire its dbs up
        for keyword, build, configured_name, label in (
            ("db_vector_connector", build_vector_db, agent_config.db_vector, "vector db"),
            ("db_text_connector", build_text_db, agent_config.db_text, "text db"),
            ("embeddings_connector", build_embeddings, embeddings, "embeddings"),
        ):
            kwargs[keyword] = (
                optional(build, f"{label} {configured_name!r}", name=configured_name, configs=configs)
                if configured_name
                else None
            )
    if labels and agent_class.__name__ == "ClassifierAgent":
        kwargs["labels"] = list(labels)
    elif labels:
        logger.warning(f"labels only constrain a classifier agent; {agent_class.__name__} ignores them.")

    return agent_class(
        name=name,
        agent_config=agent_config,
        llm=llm,
        current_filename=config_dir,
        substitute_llm=fallback or None,
        **kwargs,
    )


def build_agents(
    names: Iterable[str] | None = None,
    configs: Configs | str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build several agents at once, keyed by name — every agent in the config unless `names` says which.

    This is what a pipeline is assembled out of: the agents a run needs, built once, then called in
    whatever order the steps put them in.
    """
    configs, _ = as_configs(configs)
    if names is None:
        if not configs:
            raise ValueError("no config to read agents from: pass `configs`, or the `names` to build.")
        names = list(configs.agent_configs)

    return {name: build_agent(name, configs, **kwargs) for name in names}
