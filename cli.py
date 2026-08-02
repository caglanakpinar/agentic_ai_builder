"""`agentic-ai`: build the pieces of an agentic pipeline from the command line.

Each command under `generate` builds one live object — an agent, a provider caller, a vector db — and
reports what it built. `-c/--config-dir` points at the directory holding the pipeline's YAML; `--name` is
then the key the object is listed under in it (an entry of `agents:`, `llms:` or `dbs:`), and any flag
given alongside overrides what the YAML says. With no `--config-dir` the flags alone have to describe the
object, which is how you try a provider or an engine out before writing any config for it.

Building is the point: a command that returns without an error has imported the driver, resolved the
credentials, connected, and — for an agent — imported every tool function and read every prompt. The
optional `--question`/`--prompt` flags go one step further and run what was built once.

This module is only the command line over `builder.factory`, which is the same layer to import when
assembling a pipeline in Python — see `agent_builder`.
"""

import json
from typing import Any

import click

from builder.factory import (
    AGENT_TYPES,
    DEFAULT_MAX_TOKENS,
    LLM_CALLERS,
    VECTOR_DBS,
    build_agent,
    build_llm,
    build_vector_db,
    optional,
)


def parse_settings(values: tuple[str, ...]) -> dict[str, Any]:
    """Parse repeated `--set key=value` flags into the constructor kwargs they stand for.

    Values are JSON-decoded when they parse as JSON, so `--set top_p=0.9`, `--set stream=true` and
    `--set stop='["\\n"]'` arrive as a float, a bool and a list rather than as three strings.
    """
    settings: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator:
            raise click.BadParameter(f"--set expects key=value, got {value!r}.")

        try:
            settings[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            settings[key.strip()] = raw

    return settings


def given(**flags: Any) -> dict[str, Any]:
    """Keep only the flags that were passed, so an unset one can't blank a configured value."""
    return {key: value for key, value in flags.items() if value not in (None, (), [])}


def report(title: str, **values: Any) -> None:
    """Print what a command built, one field per line."""
    click.secho(title, fg="green", bold=True)
    width = max((len(key) for key in values), default=0)
    for key, value in values.items():
        click.echo(f"  {key.ljust(width)}  {value}")


class BuilderGroup(click.Group):
    """Reports what the builders raise for unusable config as a plain CLI error.

    `builder.factory` signals config it cannot build from with `ValueError`, and a driver that isn't
    installed with `ImportError` — neither is a crash, so both are worth one line rather than a
    traceback. Anything else keeps its traceback, because it is a bug rather than an input problem.
    """

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (ValueError, ImportError) as error:
            raise click.ClickException(str(error))


@click.group(cls=BuilderGroup, context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Build agentic pipelines — agents, LLM callers and databases — from one YAML config."""


@cli.group(cls=BuilderGroup)
def generate() -> None:
    """Generate one piece of a pipeline, or the whole pipeline."""


@generate.command("agentic_ai")
@click.option("--config-dir", "-c", required=True, help="Directory holding the pipeline's YAML config.")
@click.option("--question", help="The question to run the pipeline against.")
@click.option("--context", default="", help="Context passed to the pipeline's first step.")
def agentic_ai(config_dir: str, question: str | None, context: str) -> None:
    """Build and run the whole pipeline described by a YAML config. (not implemented yet)"""
    # TODO: walk the `pipeline:` block — build every agent, db and orchestrator its steps reference
    # (`build_agents` already builds the agents in one call), run each step in order (honouring `loops`,
    # `next_step` and `needs_judger`), feeding each agent's output into the next through
    # `agent_outputs`, and print the final result.
    pass


@generate.command("agent")
@click.option("--name", required=True, help="Agent name; also the key it is looked up under in `agents:`.")
@click.option(
    "--config-dir",
    "-c",
    default=".",
    show_default=True,
    help="Directory holding the YAML config, and the root the agent's prompt and tool paths resolve against.",
)
@click.option("--type", "agent_type", help=f"Agent role. One of: {', '.join(sorted(AGENT_TYPES))}.")
@click.option("--prompt", help="Directory of `.md` prompts for this agent, relative to --config-dir.")
@click.option("--responsiblity-prompt", help="Inline responsibility prompt, when the agent has no prompt directory.")
@click.option("--tool", "tools", multiple=True, help="Tool this agent may call, by name from `tools:`. Repeatable.")
@click.option("--mcp-server", "mcp_servers", multiple=True, help="MCP server to connect for this agent. Repeatable.")
@click.option("--db-vector", help="Vector db this agent retrieves from, by name from `dbs:`.")
@click.option("--db-text", help="Text db this agent fetches documents from, by name from `dbs:`.")
@click.option("--db-sql", help="SQL db this agent queries, by name from `dbs:`.")
@click.option("--embeddings", help="Embeddings config this agent embeds its question with, by name.")
@click.option("--label", "labels", multiple=True, help="Allowed label for a classifier agent. Repeatable.")
@click.option("--llm", "llm_name", help="LLM that backs this agent, by name from `llms:`.")
@click.option("--substitute-llm", help="LLM to fall back to when the primary call fails, by name.")
@click.option("--provider", help="Provider serving the model, when it can't be inferred from the model id.")
@click.option("--model-name", help="Model id to call, instead of (or on top of) the configured `llms:` entry.")
@click.option("--api-key", help="Provider API key, or the name of the environment variable holding it.")
@click.option("--temperature", type=float, help="Sampling temperature for this agent's calls.")
@click.option("--max-tokens", type=int, help="Max tokens this agent's calls may generate.")
@click.option("--set", "settings", multiple=True, help="Extra provider option as key=value. Repeatable.")
@click.option("--question", help="Run the agent once against this question after building it.")
@click.option("--context", default="", help="Context passed to the agent when --question is given.")
def agent(
    name: str,
    config_dir: str,
    agent_type: str | None,
    prompt: str | None,
    responsiblity_prompt: str | None,
    tools: tuple[str, ...],
    mcp_servers: tuple[str, ...],
    db_vector: str | None,
    db_text: str | None,
    db_sql: str | None,
    embeddings: str | None,
    labels: tuple[str, ...],
    llm_name: str | None,
    substitute_llm: str | None,
    provider: str | None,
    model_name: str | None,
    api_key: str | None,
    temperature: float | None,
    max_tokens: int | None,
    settings: tuple[str, ...],
    question: str | None,
    context: str,
) -> None:
    """Generate one agent from an `AgentConfigs`, built out of these flags and the `agents:` config.

    The agent's tools are imported and its prompts read while it is built, so a command that returns
    without an error is an agent that can run. `--question` runs it once and prints what it produced.
    """
    built = build_agent(
        name,
        config_dir,
        llm=llm_name,
        substitute_llm=substitute_llm,
        embeddings=embeddings,
        labels=labels,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        settings=parse_settings(settings),
        **given(
            type=agent_type,
            prompt=prompt,
            responsiblity_prompt=responsiblity_prompt,
            tools=list(tools),
            mcp_servers=list(mcp_servers),
            db_vector=db_vector,
            db_text=db_text,
            db_sql=db_sql,
        ),
    )

    config = built.agent_config
    report(
        f"Built agent {name}",
        type=built.type,
        agent=type(built).__name__,
        llm=f"{type(built.llm).__name__}({built.llm.model_name})",
        substitute_llm=(
            f"{type(built.substitute_llm).__name__}({built.substitute_llm.model_name})"
            if built.substitute_llm
            else "-"
        ),
        prompts=config.prompt or "-",
        tools=", ".join(built.toolbox.agent_tools) if built.toolbox else "-",
        mcp_servers=", ".join(built.mcp_servers or []) or "-",
        dbs=", ".join(filter(None, [config.db_vector, config.db_text, config.db_sql])) or "-",
    )

    if question:
        click.echo()
        click.echo(built.run(question=question, context=context, agent_outputs={}))


@generate.command("llm_caller")
@click.option("--name", help="LLM name to look up in the `llms:` config; flags below override its fields.")
@click.option("--config-dir", "-c", help="Directory holding the YAML config. Omit to build from flags alone.")
@click.option("--provider", help=f"Provider to call. One of: {', '.join(sorted(LLM_CALLERS))}.")
@click.option("--model-name", help="Model id to call, optionally prefixed with `provider/`.")
@click.option("--api-key", help="Provider API key, or the name of the environment variable holding it.")
@click.option("--temperature", type=float, help="Sampling temperature.")
@click.option("--max-tokens", type=int, help=f"Max tokens to generate. [default: {DEFAULT_MAX_TOKENS}]")
@click.option("--type", "llm_type", help="Role this caller plays: generator, retriever, tool caller, tool generator.")
@click.option("--mcp-server", "mcp_servers", multiple=True, help="MCP server to connect for this caller. Repeatable.")
@click.option("--set", "settings", multiple=True, help="Extra provider option as key=value, e.g. --set top_p=0.9. Repeatable.")
@click.option("--prompt", help="Run one generation against this prompt after building the caller.")
def llm_caller(
    name: str | None,
    config_dir: str | None,
    provider: str | None,
    model_name: str | None,
    api_key: str | None,
    temperature: float | None,
    max_tokens: int | None,
    llm_type: str | None,
    mcp_servers: tuple[str, ...],
    settings: tuple[str, ...],
    prompt: str | None,
) -> None:
    """Generate one provider caller (Claude, OpenAI, Google, Grok, Ollama, Mistral, Hugging Face).

    The provider is taken from `--provider`, or inferred from the model id — a `provider/` prefix first,
    then the id itself, so `--model-name claude-sonnet-5` is enough on its own. Building it resolves the
    API key and constructs the SDK client; `--prompt` then runs one generation through it.
    """
    caller = build_llm(
        name=name,
        configs=config_dir,
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        type=llm_type,
        mcp_servers=mcp_servers,
        settings=parse_settings(settings),
    )

    report(
        f"Built LLM caller {name or caller.model_name}",
        caller=type(caller).__name__,
        model=caller.model_name,
        type=caller.type,
        temperature=caller.temperature,
        max_tokens=caller.max_tokens,
        mcp_servers=", ".join(caller.mcp_servers or []) or "-",
    )

    if prompt:
        click.echo()
        click.echo(caller._call(prompt))


@generate.command("vector_db")
@click.option("--name", help="Db name to look up in the `dbs:` config; flags below override its fields.")
@click.option("--config-dir", "-c", help="Directory holding the YAML config. Omit to build from flags alone.")
@click.option("--db", help=f"Engine to connect with. One of: {', '.join(sorted(VECTOR_DBS))}.")
@click.option("--host", help="Server host, for client-server engines.")
@click.option("--port", type=int, help="Server port, for client-server engines.")
@click.option("--url", help="Full server URL; takes precedence over --host/--port.")
@click.option("--api-key", help="API key, or the name of the environment variable holding it.")
@click.option("--path", help="Storage directory or index file, for embedded engines.")
@click.option("--collection-name", help="Collection/index the connector reads and writes. [default: default]")
@click.option("--dimension", type=int, help="Embedding width; required by engines that build the index up front.")
@click.option("--metric", type=click.Choice(["cosine", "l2", "ip"]), help="Similarity metric. [default: cosine]")
@click.option("--set", "settings", multiple=True, help="Extra driver option as key=value, e.g. --set nlist=100. Repeatable.")
def vector_db(
    name: str | None,
    config_dir: str | None,
    db: str | None,
    host: str | None,
    port: int | None,
    url: str | None,
    api_key: str | None,
    path: str | None,
    collection_name: str | None,
    dimension: int | None,
    metric: str | None,
    settings: tuple[str, ...],
) -> None:
    """Generate one vector db connector (FAISS, Chroma, Qdrant, Pinecone, Weaviate, Milvus, LanceDB).

    Building it imports the driver, opens or creates the collection, and connects — so this is also how
    you check a `dbs:` entry is reachable before an agent depends on it. The vectors it already holds are
    reported once it is up.
    """
    connector = build_vector_db(
        name,
        config_dir,
        settings=parse_settings(settings),
        **given(
            db=db,
            host=host,
            port=port,
            api_key=api_key,
            path=path,
            url=url,
            collection_name=collection_name,
            dimension=dimension,
            metric=metric,
        ),
    )

    report(
        f"Built vector db {connector.name}",
        connector=type(connector).__name__,
        engine=connector.db,
        collection=connector.collection_name,
        location=connector.url or connector.host or connector.path or "-",
        metric=connector.metric,
        dimension=connector.dimension or "-",
        vectors=optional(connector.count, "vector count") if connector.client is not None else "-",
    )


if __name__ == "__main__":
    cli()
