from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import importlib
import os
import re
import yaml


ENV_VAR_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")  # what a configured environment variable name looks like

# The registries and defaults shipped with the package — which provider, engine and agent-type names
# exist, and what class each one is built from. Read once here rather than per `Configs`: it is package
# data, and it does not change under a running process.
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def read_default_configs() -> dict[str, Any]:
    """Read `default_config.yaml`, the defaults every builder resolves against."""
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"the package default config is missing from {DEFAULT_CONFIG_PATH}.")

    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}


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


def resolve_secret(value: str | None, owner: str, field: str = "api_key", required: bool = True) -> str | None:
    """Resolve a configured secret as either the environment variable it names, or the literal value.

    A value spelled like an environment variable — upper case, digits and underscores — is only ever a
    name: no provider issues keys that look like that. So when the variable behind it is unset, that is a
    missing key rather than a key that happens to be spelled `CLAUDE`, and it is refused here. Passing it
    through instead sends the name itself as the credential, and the provider answers with a 401 that
    says nothing about which variable was empty.

    Args:
        value: What the config holds — an environment variable name, or the secret itself.
        owner: What the secret belongs to, for the error message (a model or db name).
        field: The config key being resolved, for the error message.
        required: Whether a missing value is an error. Optional for dbs that need no credential.
    """
    if value and os.getenv(value):
        return os.getenv(value)

    if value and ENV_VAR_NAME.match(value):
        raise ValueError(
            f"{owner}: `{field}` names the environment variable {value}, which is not set. "
            f"Export it (export {value}=...) or put the secret itself in the config."
        )

    if not value and required:
        raise ValueError(
            f"{owner}: no `{field}` configured — give the secret, or the name of the environment "
            "variable holding it."
        )

    return value


@dataclass
class LLMConfigs:
    model_name: str
    # Left unset unless the config names one: the newest Claude models reject `temperature` outright
    # (Opus 5, Opus 4.8/4.7, Fable 5) or accept only their default (Sonnet 5), so a sampling temperature
    # is a per-model choice rather than something to send on every request.
    temperature: float | None
    max_tokens: int
    api_key: str
    tools: list[dict[str, str]] | None = None
    type: str = "generator" # options are "generator" or "retriever", "tool caller", "tool generator"
    mcp_servers: list[str] | None = None
    # Provider-specific options passed to the caller's constructor — whatever that provider declares as
    # its own params: `provider` for the Hugging Face router, `thinking_config` for Gemini, `top_p` for
    # most. Anything here is the provider's vocabulary, not this config's, so nothing validates it.
    settings: dict[str, Any] | None = None


@dataclass
class AgentConfigs:
    responsiblity_prompt: str
    tools: list[dict[str, str]] | None = None
    type: str = "generator" # options are "generator" or "retriever", "tool caller", "tool generator"
    mcp_servers: list[str] | None = None
    llms: dict[str, LLMConfigs] | None = None
    db_vector: str | None = None  # name of the vector db this agent retrieves from, e.g. "ds_knowledge_db"
    db_text: str | None = None  # name of the text db this agent searches, e.g. "ds_knowledge_db_text"
    db_sql: str | None = None  # name of the tabular db this agent queries, e.g. "authentication_db"
    embedding: str | None = None  # name of the embeddings caller this agent turns its query into a vector with
    prompt: str | None = None  # directory (relative to Configs.current_dir) containing one file per prompt, e.g. "prompts/rag_problem_thinker_agent"
    prompt_path: str | None = None  # a single .md file to use as this agent's prompt, when it has only one
    thresholds: dict[str, Any] | None = None  # the bars a judger holds work to, e.g. {"min_roc_auc": 0.7}
    # The agent(s) whose output this one works from — one name or a list. Their outputs are what
    # `{agent_output}` renders as, so a prompt asks for "what came before" instead of naming each
    # upstream agent itself, and rewiring the pipeline is a config change rather than a prompt edit.
    dependency_agent: str | list[str] | None = None
    arguments: list[str] = field(default_factory=list)  # prompt names discovered in `prompt`, filled in by BasePrompt.prompt_configer

@dataclass
class VectorDBConfigs:
    name: str  # db name as referenced by agents, e.g. "ds_knowledge_db"
    db: str  # engine to connect with: "faiss", "chroma", "qdrant", "pinecone", "weaviate", ...
    type: str = "vector"  # db category `db_configer` routes on; see DB_CONFIGS
    host: str | None = None
    port: int | None = None
    api_key: str | None = None  # literal key, or the name of an env var holding it
    path: str | None = None  # storage directory/file for embedded engines (FAISS, Chroma, Qdrant, LanceDB)
    url: str | None = None  # full server URL; when set it takes precedence over host/port
    collection_name: str = "default"  # collection/index/class the connector reads and writes
    dimension: int | None = None  # embedding width; required by engines that build the index up front
    metric: str = "cosine"  # similarity metric: "cosine", "l2", or "ip"


@dataclass
class SQLDBConfigs:
    name: str  # db name as referenced by agents, e.g. "ds_knowledge_db"
    db: str  # engine to connect with: "postgresql", "mysql", "bigquery", "sqlite", "snowflake", ...
    type: str = "sql"  # db category `db_configer` routes on; see DB_CONFIGS
    host: str | None = None
    port: int | None = None
    database: str | None = None  # database/schema (BigQuery: dataset) to run against
    user: str | None = None
    password: str | None = None  # literal password, or the name of an env var holding it
    connection_string: str | None = None  # full DSN; when set it takes precedence over the fields above
    path: str | None = None  # file path for file-backed engines (SQLite, DuckDB) or a seed .sql script
    project: str | None = None  # cloud project id, used by BigQuery
    credentials: str | None = None  # path to a service-account JSON, or the name of an env var holding it


@dataclass
class TextDBConfigs:
    """A document store searched with a text query rather than a vector or SQL."""

    name: str  # db name as referenced by agents, e.g. "ds_knowledge_db_text"
    db: str  # engine to connect with: "elasticsearch", "opensearch", "meilisearch", "typesense", ...
    type: str = "text"  # db category `db_configer` routes on; see DB_CONFIGS
    host: str | None = None
    port: int | None = None
    api_key: str | None = None  # literal key, or the name of an env var holding it
    user: str | None = None
    password: str | None = None  # literal password, or the name of an env var holding it
    path: str | None = None  # storage directory/file for embedded engines
    url: str | None = None  # full server URL; when set it takes precedence over host/port
    collection_name: str = "default"  # index/collection the connector reads and writes
    analyzer: str | None = None  # text analyzer the engine tokenises with, e.g. "standard", "english"


@dataclass
class EmbeddingsConfigs:
    model_name: str
    api_key: str


@dataclass
class ToolConfigs:
    name: str
    description: str
    type: str  # options are "generator" or "retriever", "tool caller", "tool generator"
    llm: str | None = None
    embeddings: str | None = None
    config: dict[str, str] | None = None
    caller: str | None = None
    args: list[dict[str, str]] | None = None


@dataclass
class PipelineConfigs:
    name: str
    description: str
    tools: list[ToolConfigs]
    llm: str | None = None
    embeddings: str | None = None
    config: dict[str, str] | None = None


class Configs:
    # The package defaults, on the class so a builder can read a registry without a config directory —
    # `builder.factory` resolves against these while it is being imported. An instance carries the same
    # blocks as attributes, overlaid with anything a pipeline's own YAML redefines (see `read_yaml`);
    # these three accessors read the shipped defaults rather than one pipeline's view of them.
    default_configs: dict[str, Any] = read_default_configs()

    db_confgs: dict[str, VectorDBConfigs | SQLDBConfigs | TextDBConfigs] = {}
    llm_configs: dict[str, LLMConfigs] = {}
    embeddings_configs: dict[str, EmbeddingsConfigs] = {}
    agent_configs: dict[str, AgentConfigs] = {}
    tool_configs: dict[str, ToolConfigs] = {}

    @classmethod
    def registry(cls, name: str) -> dict[str, str]:
        """One `name -> dotted path` block of the default config, e.g. `registry("llm_callers")`."""
        block = cls.default_configs.get(name)
        if not isinstance(block, dict):
            raise KeyError(f"{name!r} is not a registry in {DEFAULT_CONFIG_PATH.name}.")

        return dict(block)

    @classmethod
    def setting(cls, name: str, default: Any = None) -> Any:
        """One value from the `defaults:` block, e.g. `setting("max_tokens")`."""
        return (cls.default_configs.get("defaults") or {}).get(name, default)

    @classmethod
    def db_connectors(cls) -> dict[str, tuple[dict[str, str], type]]:
        """Each db category -> the engines it can use, and the configs class its entries are parsed into.

        The configs classes are resolved on call rather than at import, because they are defined in this
        module: importing them by path while it is still executing would be a partial import of itself.
        """
        return {
            category: (cls.registry(entry["engines"]), load_class(entry["configs"]))
            for category, entry in (cls.default_configs.get("db_categories") or {}).items()
        }

    def __init__(self, current_filename: str):
        self.current_dir = self.resolve_dir(current_filename)
        self.read_yaml()
        self.db_configer()
        self.llm_configer()
        self.embeddings_configer()
        self.tool_configer()
        self.agent_configer()

    @staticmethod
    def resolve_dir(current_filename: str) -> Path:
        """Locate the directory holding the YAML config.

        A path that exists as given — absolute, or relative to where the command was run — is taken as
        it is, which is how the CLI points at a config directory in the user's own project. Anything
        else falls back to being read relative to this package.
        """
        given = Path(current_filename)
        if given.is_dir():
            return given

        return Path(__file__).parent / current_filename

    def read_yaml(self):
        """Lay the pipeline's own YAML over the package defaults, both as attributes on this config.

        The defaults go on first, so a pipeline that redefines a block — its own `agent_types:`, say —
        replaces the shipped one for this config rather than being merged into it.
        """
        for key, value in self.default_configs.items():
            setattr(self, key, value)

        for file in self.current_dir.glob("*.yaml"):
            with open(file, 'r') as f:
                configs = yaml.safe_load(f) or {}

            for key, value in configs.items():
                setattr(self, key, value)
            return

        raise FileNotFoundError(f"No YAML configuration file found in {self.current_dir}.")

    def db_configer(self):
        """Parse the `dbs:` block, keyed by db name, routing each entry by its `type`.

        Accepts the block either as a list of db mappings or as a name -> mapping dict. `type` is the
        category the entry is routed by — "vector", "sql", or "text" — and `db` is the engine to
        connect with within that category, e.g. type "vector" with db "chroma".
        """
        if 'dbs' in self.__dict__:
            dbs = self.dbs
            if isinstance(dbs, dict):  # name -> mapping form, where the key carries the db name
                dbs = [{'name': db_name, **db_cfg} for db_name, db_cfg in dbs.items()]

            categories = self.db_connectors()
            for db_cfg in dbs:
                if db_cfg.get('type') not in categories:
                    raise ValueError(
                        f"{db_cfg['name']}: unknown db type {db_cfg.get('type')!r}, "
                        f"expected one of {sorted(categories)}."
                    )

                _, configs = categories[db_cfg['type']]
                if not db_cfg.get('db'):
                    raise ValueError(f"{db_cfg['name']}: `db` must name the engine to connect with.")

                self.db_confgs[db_cfg['name']] = configs(
                    **{key: value for key, value in db_cfg.items() if key in configs.__dataclass_fields__}
                )

    def llm_configer(self):
        """Parse the `llms:` block (also accepted as `llm:`), keyed by the name agents reference."""
        llms = self.__dict__.get('llms', self.__dict__.get('llm'))
        if llms:
            for llm_name, llm_cfg in llms.items():
                self.llm_configs[llm_name] = LLMConfigs(
                    model_name=llm_cfg.get('model_name', llm_cfg.get('model', '')),
                    temperature=llm_cfg.get('temperature'),  # None when unset — see LLMConfigs
                    max_tokens=llm_cfg.get('max_tokens', 0),
                    api_key=llm_cfg.get('api_key', ''),
                    type=llm_cfg.get('type', 'generator'),
                    tools=llm_cfg.get('tools', None),
                    mcp_servers=llm_cfg.get('mcp_servers', None),
                    settings=llm_cfg.get('settings', None)
                )

    def embeddings_configer(self):
        if 'embeddings' in self.__dict__:
            for embedding_name, embedding_cfg in self.embeddings.items():
                self.embeddings_configs[embedding_name] = EmbeddingsConfigs(
                    model_name=embedding_cfg.get('model_name', embedding_cfg.get('model', '')),
                    api_key=embedding_cfg.get('api_key', '')
                )

    def tool_configer(self):
        """Parse the `tools:` block, keyed by tool name — the registry agents reference tools from.

        Accepts the block either as a list of tool mappings or as a name -> mapping dict.
        """
        if 'tools' in self.__dict__:
            tools = self.tools
            if isinstance(tools, dict):  # name -> mapping form, where the key carries the tool name
                tools = [{'name': tool_name, **tool_cfg} for tool_name, tool_cfg in tools.items()]

            for tool_cfg in tools:
                self.tool_configs[tool_cfg['name']] = ToolConfigs(
                    name=tool_cfg['name'],
                    description=tool_cfg.get('description', ''),
                    type=tool_cfg.get('type', 'tool caller'),
                    llm=tool_cfg.get('llm', None),
                    embeddings=tool_cfg.get('embeddings', None),
                    config=tool_cfg.get('config', None),
                    caller=tool_cfg.get('caller', None),
                    args=tool_cfg.get('args', None)
                )

    def agent_configer(self):
        if 'agents' in self.__dict__:
            for agent_name, agent_cfg in self.agents.items():
                self.agent_configs[agent_name] = AgentConfigs(
                    responsiblity_prompt=agent_cfg.get('responsiblity_prompt', ''),
                    tools=agent_cfg.get('tools', None),
                    type=agent_cfg.get('type', 'generator'),
                    mcp_servers=agent_cfg.get('mcp_servers', None),
                    llms=agent_cfg.get('llms', None),
                    db_vector=agent_cfg.get('db_vector', None),
                    db_text=agent_cfg.get('db_text', None),
                    embedding=agent_cfg.get('embedding', None),
                    db_sql=agent_cfg.get('db_sql', None),
                    prompt=agent_cfg.get('prompt', None),
                    prompt_path=agent_cfg.get('prompt_path', None),
                    thresholds=agent_cfg.get('thresholds', None),
                    dependency_agent=agent_cfg.get('dependency_agent', None)
                )