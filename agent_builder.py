"""agent_builder: build agentic pipelines from a YAML config, in Python or from the command line.

This is the package's public surface. Everything the `agentic-ai` command builds is a function call
here, against the same config, so a pipeline assembled in Python is the pipeline the CLI builds:

    from agent_builder import build_agent, build_llm, build_vector_db, load_configs

    configs = load_configs("my_pipeline")           # read the YAML once, reuse it everywhere

    agent = build_agent("triage_agent", configs)    # tools imported, prompts located, LLM constructed
    print(agent.run(question="the login page 500s", context="", agent_outputs={}))

    caller = build_llm("main_llm", configs)         # or build_llm(model_name="claude-sonnet-5", api_key=...)
    db = build_vector_db("knowledge_db", configs)   # driver imported, collection opened

Every builder takes its values from the config, from keyword overrides, or from both — an override always
wins over what the YAML says — and raises `ValueError` for config it cannot build from. See
`builder.factory` for the full signatures.

The classes behind those builders are exported too (`WorkerAgent`, `ClaudeLLM`, `ChromaDB`, …), but they
are resolved on first use rather than at import: a provider SDK or a db driver is only imported once
something actually asks for it, so `import agent_builder` stays cheap and needs no driver installed.
"""

from typing import Any

from builder.factory import (
    AGENT_TYPES,
    DB_CONNECTORS,
    EMBEDDINGS,
    LLM_CALLERS,
    SQL_DBS,
    TEXT_DBS,
    VECTOR_DBS,
    as_configs,
    build_agent,
    build_agents,
    build_db,
    build_embeddings,
    build_llm,
    build_sql_db,
    build_text_db,
    build_vector_db,
    load_configs,
)
from uilts.configs import (
    AgentConfigs,
    Configs,
    EmbeddingsConfigs,
    LLMConfigs,
    PipelineConfigs,
    SQLDBConfigs,
    TextDBConfigs,
    ToolConfigs,
    VectorDBConfigs,
)
from uilts.logger import get_logger, logger

__version__ = "0.1.0"

# Classes exported by name but imported only when one is first used — the registries already say where
# each provider caller, connector and agent class lives, so the base classes are all that need listing.
LAZY_EXPORTS: dict[str, str] = {
    "BaseAgent": "builder.agents.BaseAgent",
    "BasePrompt": "builder.prompts.BasePrompt",
    "Tool": "builder.tools.Tool",
    "ToolBox": "builder.tools.ToolBox",
    "BaseLLM": "models.llms.BaseLLM",
    "BaseEmbeddings": "models.embeddings.BaseEmbeddings",
    "BaseVectorDB": "db_connector.vector.BaseVectorDB",
    "BaseTextDB": "db_connector.text.BaseTextDB",
    "BaseSQLDB": "db_connector.tabular.BaseSQLDB",
    **{
        path.rsplit(".", 1)[1]: path
        for registry in (AGENT_TYPES, LLM_CALLERS, EMBEDDINGS, VECTOR_DBS, TEXT_DBS, SQL_DBS)
        for path in registry.values()
    },
}


def __getattr__(name: str) -> Any:
    """Import an exported class on first use, so importing this module pulls in no SDK or driver."""
    if name not in LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from builder.factory import load_class

    value = load_class(LAZY_EXPORTS[name])
    globals()[name] = value  # cache it, so the import cost is paid once
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(LAZY_EXPORTS))


__all__ = [
    # builders
    "build_agent",
    "build_agents",
    "build_db",
    "build_embeddings",
    "build_llm",
    "build_sql_db",
    "build_text_db",
    "build_vector_db",
    "load_configs",
    "as_configs",
    # config
    "Configs",
    "AgentConfigs",
    "LLMConfigs",
    "EmbeddingsConfigs",
    "ToolConfigs",
    "PipelineConfigs",
    "VectorDBConfigs",
    "TextDBConfigs",
    "SQLDBConfigs",
    # registries
    "AGENT_TYPES",
    "LLM_CALLERS",
    "EMBEDDINGS",
    "VECTOR_DBS",
    "TEXT_DBS",
    "SQL_DBS",
    "DB_CONNECTORS",
    # logging
    "logger",
    "get_logger",
    *sorted(LAZY_EXPORTS),
]
