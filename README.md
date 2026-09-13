# agent_builder

Build agentic AI pipelines from a YAML file instead of from glue code.

You declare what the pipeline is made of — which models it calls, which databases it retrieves from,
which tools its agents may run, and which agents run in what order — and `agent_builder` constructs the
live objects behind those names: provider callers, db connectors, tool functions imported from your own
modules, and agents that render their prompts and call their LLM.

The point of routing everything through one config is that the pieces stay swappable. Every LLM
provider is called through the same `_call(prompt, **kwargs)`, every vector db through the same
`upsert / query / delete / count`, so moving an agent from Claude to Gemini, or from FAISS to Qdrant, is
a line of YAML rather than a rewrite of the code that calls it.

```
                 configs.yaml
                      │
   ┌──────────┬───────┴───────┬─────────────┐
   ▼          ▼               ▼             ▼
  llms    embeddings         dbs          tools
   │          │               │             │
   └──────────┴───────┬───────┴─────────────┘
                      ▼
                    agents  ──►  orchestrators  ──►  pipeline
```

---

## Install

```bash
poetry install                           # core: config, agents, prompts, tools, LLM providers
poetry install --extras "faiss postgres" # add just the db drivers you use
poetry install --extras "all"            # every db driver
```

Database drivers are optional and imported lazily, so using Chroma never requires installing Pinecone.
The extras are listed in [pyproject.toml](pyproject.toml).

Installing gives you both ways in: `import agent_builder` from Python, and the `agentic-ai` command on
your PATH. Without installing, both still work from the repository root — `import agent_builder` and
`python cli.py ...`.

---

## Quick start

Write a config directory holding one `.yaml` file:

```yaml
# my_pipeline/configs.yaml
llms:
  main_llm:
    model: "claude-sonnet-5"
    api_key: ANTHROPIC_API_KEY   # the env var's name, or paste the key itself
    max_tokens: 2048
    temperature: 0.2

dbs:
  - name: "knowledge_db"
    type: "vector"
    db: "chroma"
    path: "my_pipeline/chroma"

tools:
  - name: "echo_tool"
    description: "Echo a message back."
    caller: "my_pipeline.tools.echo_tool"
    args:
      - name: "message"
        type: "str"
        description: "What to echo."

agents:
  triage_agent:
    type: "classifier"
    llm: "main_llm"
    prompt: "prompts/triage_agent"
    tools:
      - "echo_tool"
```

Then build the pieces — from Python:

```python
from agent_builder import build_agent, build_llm, build_vector_db, load_configs

configs = load_configs("my_pipeline")            # read the YAML once, reuse it everywhere

agent = build_agent("triage_agent", configs)     # tools imported, prompts located, LLM constructed
print(agent.run(question="the login page 500s", context="", agent_outputs={}))
```

or from the command line, which drives exactly the same builders:

```bash
agentic-ai generate llm_caller -c my_pipeline --name main_llm
agentic-ai generate vector_db  -c my_pipeline --name knowledge_db
agentic-ai generate agent      -c my_pipeline --name triage_agent --question "the login page 500s"
```

Either way the real object is built: the API key is resolved and the SDK client constructed, the db
driver imported and the collection opened, every tool function imported and every prompt read. Nothing
that returns without an error is a piece that can't run.

---

## The config file

`Configs` reads the first `.yaml` file in the directory you point it at, and parses these blocks.

### `llms:`

Named provider callers. `model` is the provider's own model id, optionally prefixed with `provider/`
when the id alone doesn't say who serves it. `api_key` is read as an environment variable name first,
and used as a literal key when no such variable is set.

```yaml
llms:
  main_llm:
    model: "claude-sonnet-5"     # or "google/gemini-2.5-pro", "gpt-4o", "mistral-large-latest"
    api_key: ANTHROPIC_API_KEY
    temperature: 0.2
    max_tokens: 2048
    type: "generator"            # generator | retriever | tool caller | tool generator
```

| Provider | Names accepted | Caller |
| --- | --- | --- |
| Anthropic | `claude`, `anthropic` | `ClaudeLLM` — the only caller that forwards `mcp_servers` |
| OpenAI | `openai` | `OpenAILLM` |
| Google | `google`, `gemini` | `GoogleLLM` |
| xAI | `grok`, `xai` | `GrokLLM` |
| Mistral | `mistral` | `MistralLLM` |
| Ollama (local) | `ollama` | `OllamaLLM` |
| Hugging Face | `huggingface`, `hf` | `HuggingFaceInferenceLLM` |
| Hugging Face (local) | `huggingface_local`, `hf_local` | `HuggingFaceLocalLLM`: downloads the repo and runs it with `transformers`. A PEFT adapter repo is loaded over its base model (needs `pip install peft`). Tool calls are parsed from the model's `<tool_call>` replies. No key needed for public repos |
| Layer-LoRA adapter (local) | `layer_lora`, `hf_layer_lora` | `LayerLoraAdapterHuggingFaceLLM`: the same in-process load for a layer-scoped LoRA fine-tune. Defaults `system` to the prompt that adapter was trained under, resolves a training run directory (`outputs/sft-layer-lora`) to the adapter inside it, and refuses to start when the adapter's update is zero — which would silently serve the base model |

Provider-specific options (`top_p`, `thinking`, `response_format`, `stop`, …) are declared per caller in
[models/llms.py](models/llms.py), and can be set with `--set key=value` on the CLI.

### `embeddings:`

The models a RAG agent turns its question into a vector with. OpenAI, Google, Hugging Face, Mistral and
Ollama are supported, plus `local` — see [models/embeddings.py](models/embeddings.py).

```yaml
embeddings:
  rag_embeddings:
    model: "openai/text-embedding-3-small"
    api_key: OPENAI_API_KEY
```

`local/hashing-<width>` is the one that calls nobody: it hashes words and word pairs into a vector of
that width, in-process, with no key and no download. It matches wording rather than meaning, so it is a
way to get retrieval running — offline, in CI, before a provider is chosen — not a replacement for a
trained model. **There is no Claude option**: the Anthropic API has no embeddings endpoint, so an
Anthropic key cannot serve retrieval however the config is written.

### `dbs:`

Databases, keyed by the name agents reference. `type` is the category (`vector`, `text`, `sql`), and
`db` is the engine within it.

```yaml
dbs:
  - name: "knowledge_db"
    type: "vector"
    db: "chroma"
    path: "my_pipeline/chroma"
    collection_name: "docs"
    dimension: 1536        # required by engines that build the index up front
    metric: "cosine"       # cosine | l2 | ip
  - name: "knowledge_docs"
    type: "text"
    db: "elasticsearch"
    url: "http://localhost:9200"
  - name: "app_db"
    type: "sql"
    db: "postgresql"
    host: "localhost"
    database: "app"
```

| Category | Engines |
| --- | --- |
| `vector` | `faiss`, `chroma`, `qdrant`, `pinecone`, `weaviate`, `milvus`, `lancedb` |
| `text` | `chroma`, `elasticsearch`, `opensearch`, `meilisearch`, `typesense` |
| `sql` | `postgresql`, `mysql`, `sqlite`, `duckdb`, `snowflake`, `redshift`, `bigquery` |

Vector and text dbs are the two halves of one retrieval flow: the vector db runs the similarity search
and returns ids, and the text db turns those ids into the documents an agent puts in its prompt. Chroma
appears in both categories and does a different job in each — as a vector db it searches, as a text db it
is a lookup store opened with no embedding function at all, which is what a knowledge base that lives in
a directory rather than behind a search server looks like.

### `tools:`

The registry agents pick their tools from. `caller` points at the Python function to run, either as the
full dotted path to the function or as a module whose function is named after the tool. `args` becomes
the JSON Schema the provider validates the model's tool calls against, rendered into whichever dialect
that agent's provider speaks.

Tools are **executed**, not merely offered: every caller runs the ask → execute → answer loop until the
model stops asking. The loop itself lives once, in `BaseAgent.run_with_tools`; a provider supplies only
four translation methods (`converse`, `read_turn`, `assistant_turn`, `tool_result_turns`). Three dialects
cover them all — Anthropic, the OpenAI-compatible one shared by OpenAI/Grok/Ollama/Mistral/Hugging Face,
and Gemini's `function_call`/`function_response` parts.

```yaml
tools:
  - name: "data_ingestion_tool"
    description: "Ingest and preprocess a dataset."
    caller: "my_pipeline.tools.data_ingestion"
    args:
      - name: "data_source"
        type: "str"                 # str | int | float | bool | list | dict
        description: "Where to read the data from."
      - name: "preprocessing_steps"
        type: "list"
        description: "Steps to apply."
        required: false
```

A tool whose function can't be imported is logged and left out, so the model is only ever offered tools
that can actually run.

### `agents:`

```yaml
agents:
  triage_agent:
    type: "classifier"               # picks the class that runs this role
    llm: "main_llm"                  # a name from `llms:`
    substitute_llm: "backup_llm"     # called when the primary one raises
    prompt: "prompts/triage_agent"   # directory of .md prompts, relative to the config dir
    tools:
      - "echo_tool"
    mcp_servers:
      - "mc_server_1"
    db_vector: "knowledge_db"        # for a rag agent: which db it searches for the nearest documents
    db_text: "knowledge_docs"        # for a rag agent: which db turns the ids it found into documents
    embedding: "rag_embeddings"      # for a rag agent: what it turns its question into a vector with
```

A retrieving agent needs all three: `embedding` embeds the question, `db_vector` answers with the ids of
the nearest documents, and `db_text` turns those ids into the documents themselves. With any of them
missing or unreachable the agent still runs — it generates over the context it was given and logs what
was missing — so a pipeline works before its knowledge base is filled.

| `type` | Class | What it does |
| --- | --- | --- |
| `generator`, `worker` | `WorkerAgent` | Does the task work, calling tools and MCP servers |
| `judger` | `JudgerAgent` | Evaluates another agent's output and returns a verdict |
| `classifier` | `ClassifierAgent` | Assigns the question to one of a fixed set of labels |
| `planner`, `thinker` | `PlannerAgent` | Breaks a question into an ordered plan for others to run |
| `rag`, `rag_builder`, `retriever` | `RAGBuilderAgent` | Retrieves from the dbs, then generates over what it found |

### Prompts

`prompt:` names a **directory** holding one `.md` file per prompt, and each agent class reads the file
its role is named after — `worker.md`, `judger.md`, `classifier.md`, `planner.md`, `rag_builder.md`.

Inside a prompt, `{question}` and `{context}` are filled from the call, and `{any_agent_name}` is filled
with that agent's output — which is how one step's result feeds the next.

```markdown
<!-- prompts/triage_agent/classifier.md -->
Classify this request into exactly one label: bug, feature, question.

Request: {question}
What the planner found: {rag_problem_thinker_agent}
```

### `orchestrators:` and `pipeline:`

The step order, the sub-agents each orchestrator groups, and which steps need a judger. These blocks are
read by `generate agentic_ai`, which is **not implemented yet** — see [Status](#status).

A worked example of every block above is in
[benchmarks/agentic_configurations.yaml](benchmarks/agentic_configurations.yaml) — see
[the benchmark](#benchmark) below.

---

## Benchmark

[benchmarks/](benchmarks/) is a complete pipeline pointed at a real problem: predicting which
subscription customers churn, a binary classification task on a dataset generated there. Thirteen agents
— eight doing the work, five judging it — and 24 tools that actually compute what they claim. Each stage
is gated by a judge holding it to numeric thresholds declared in the YAML, and every threshold is
evaluated twice: the judge argues about it, and the runner does the arithmetic against what the tools
measured.

```bash
python benchmarks/dataset.py          # write the dataset (deterministic)
python benchmarks/run_benchmark.py    # build the workflow, run the tool chain, render every prompt
python benchmarks/run_benchmark.py --live   # ...and call the models
```

The first two need no API key: the workflow is still built for real and the tools still run, so the
numbers it prints are measured rather than described. That is what makes it a benchmark — the agents'
claims can be checked against ground truth computed with no model in the loop (5-fold ROC AUC 0.759,
holdout 0.775, against a majority-class baseline that scores 80% accuracy by predicting nobody churns).

[benchmarks/README.md](benchmarks/README.md) has the full picture.

---

## Use it from Python

Everything the CLI builds is a function call on the `agent_builder` module, against the same config — so
a pipeline assembled in Python is the pipeline the command line builds.

```python
from agent_builder import (
    load_configs,       # read a config directory into a Configs
    build_agent,        # one agent, tools imported and prompts located
    build_agents,       # several at once, keyed by name
    build_llm,          # one provider caller
    build_embeddings,   # one embeddings caller
    build_vector_db,    # one vector db connector, connected
    build_text_db,      # one text db connector
    build_sql_db,       # one SQL db connector
)
```

### The shape every builder shares

```python
build_x(name, configs, **overrides)
```

- **`name`** — the key the object is listed under in the YAML (`agents:`, `llms:`, `dbs:`, `embeddings:`).
- **`configs`** — a config directory (`"my_pipeline"`) or an already-read `Configs`. Reading it once and
  passing it around avoids re-parsing the YAML for every piece.
- **`**overrides`** — any field of that object's config, overriding what the YAML says. An override
  always wins; anything not overridden falls back to the configured entry.
- **`settings={...}`** — provider- or driver-specific options passed to the constructor (`top_p`,
  `thinking`, `nlist`, `prefer_grpc`, …), kept separate from the config fields so a typo in one is
  caught rather than silently ignored.

Both halves are optional in the other direction, too: with no `configs`, the keywords alone describe the
object, which is how you try something out before writing any config for it.

```python
configs = load_configs("my_pipeline")

agent = build_agent("triage_agent", configs)                      # exactly as configured
agent = build_agent("triage_agent", configs, model_name="claude-opus-4-5")  # same agent, different model
agent = build_agent("triage_agent", configs, type="judger", tools=["echo_tool"])

caller = build_llm("main_llm", configs)                           # from the config
caller = build_llm(model_name="claude-sonnet-5", api_key="ANTHROPIC_API_KEY",
                   max_tokens=2048, settings={"top_p": 0.9})      # from keywords alone

db = build_vector_db("knowledge_db", configs)                     # connected, ready to query
db = build_vector_db(db="faiss", path="./scratch.index", dimension=768)
```

### Running an agent

An agent's `run` takes the question, the context, and the outputs of the agents that already ran — which
is how one agent's answer becomes another's input:

```python
agents = build_agents(["planner_agent", "triage_agent"], configs)

outputs = {}
for name, agent in agents.items():
    outputs[name] = agent.run(question="the login page 500s", context="", agent_outputs=outputs)

print(outputs["triage_agent"])
```

Inside `prompts/triage_agent/classifier.md`, `{planner_agent}` is then filled with what the planner
produced. Tools work the same way from either side — the agent offers them to the model, and runs
whichever one it asks for:

```python
agent.toolbox.schemas_for(agent.llm)          # the tools as this provider wants them declared
agent.call_tool("echo_tool", {"message": "hi"})  # run the one the model asked for
```

### Building the pieces yourself

The callers and connectors are useful on their own — every LLM behind one `_call`, every vector db behind
one `upsert / query / delete / count`:

```python
caller = build_llm("main_llm", configs)
answer = caller._call("summarise this incident", temperature=0.0)

db = build_vector_db("knowledge_db", configs)
db.upsert(ids=["doc-1"], vectors=[[0.1, 0.2, ...]], documents=["..."])
matches = db.query(vector=[0.1, 0.2, ...], top_k=5)   # best match first
```

### Errors, and what gets imported when

Config a builder can't build from raises `ValueError`; a driver that isn't installed raises
`ImportError`. Neither depends on the CLI, so both are handled the usual way:

```python
try:
    db = build_vector_db("knowledge_db", configs)
except ValueError as error:      # e.g. "db 'app_db' is a 'sql' db, not a vector one."
    ...
except ImportError as error:     # e.g. chromadb isn't installed
    ...
```

`import agent_builder` pulls in no provider SDK and no db driver — the classes it exports (`WorkerAgent`,
`ClaudeLLM`, `ChromaDB`, …) are resolved the first time one is used, so you only pay for what you build.
The modules underneath can also be imported directly when you want a specific class:

```python
from builder.factory import build_agent          # the same builders, without the facade
from builder.agents import WorkerAgent, JudgerAgent
from models.llms import ClaudeLLM
from db_connector.vector import ChromaDB
```

---

## CLI

```
agentic-ai generate agentic_ai   build and run a whole pipeline from its YAML  (not implemented yet)
agentic-ai generate agent        build one agent from an AgentConfigs
agentic-ai generate llm_caller   build one provider caller
agentic-ai generate vector_db    build and connect one vector db connector
```

Every command takes its values from the YAML (`-c/--config-dir` plus `--name`, the key the object is
listed under), from flags, or from both — **a flag always overrides what the YAML says**. With no
`--config-dir`, the flags alone have to describe the object, which is how you try a provider or an
engine out before writing any config for it.

`--set key=value` passes a provider- or driver-specific option through to the constructor, and is
repeatable. Values are JSON-decoded when they parse as JSON, so `--set top_p=0.9` arrives as a float and
`--set stop='["\n"]'` as a list.

### `generate llm_caller`

```bash
# from the config
agentic-ai generate llm_caller -c my_pipeline --name main_llm

# from flags only — the provider is inferred from the model id
agentic-ai generate llm_caller \
    --model-name claude-sonnet-5 --api-key ANTHROPIC_API_KEY --max-tokens 2048 --set top_p=0.9

# build it, then run one generation through it
agentic-ai generate llm_caller -c my_pipeline --name main_llm --prompt "say hi"
```

`--provider · --model-name · --api-key · --temperature · --max-tokens · --type · --mcp-server · --set · --prompt`

### `generate vector_db`

```bash
agentic-ai generate vector_db -c my_pipeline --name knowledge_db
agentic-ai generate vector_db --db faiss --name scratch --path ./scratch.index --dimension 768
agentic-ai generate vector_db --db qdrant --url http://localhost:6333 --collection-name docs --set prefer_grpc=true
```

Reports the engine, collection, location, metric and how many vectors it already holds — so it doubles
as the check that a `dbs:` entry is reachable before an agent depends on it.

`--db · --host · --port · --url · --api-key · --path · --collection-name · --dimension · --metric · --set`

### `generate agent`

```bash
# from the config
agentic-ai generate agent -c my_pipeline --name triage_agent

# override a field, then run the agent once
agentic-ai generate agent -c my_pipeline --name triage_agent \
    --model-name claude-opus-4-5 --question "the login page 500s"

# define an agent entirely from flags
agentic-ai generate agent -c my_pipeline --name scratch_agent --type generator \
    --model-name gpt-4o --api-key OPENAI_API_KEY --tool echo_tool --prompt prompts/scratch_agent
```

Prints the agent class, the LLM behind it, the tools that turned out to be importable, and the dbs it is
wired to. An agent naming a vector db, text db or embeddings caller gets them connected while it is
built; one that can't be built is reported as a warning and left unwired rather than failing the
command, since `RAGBuilderAgent` falls back to the caller-supplied context. An already-connected
connector can be passed to `build_agent` instead, which is how a pipeline opens a db once and shares it.

`--type · --prompt · --responsiblity-prompt · --tool · --mcp-server · --db-vector · --db-text · --db-sql ·
--embeddings · --label · --llm · --substitute-llm · --provider · --model-name · --api-key · --temperature ·
--max-tokens · --set · --question · --context`

---

## Layout

| Path | What's in it |
| --- | --- |
| [agent_builder.py](agent_builder.py) | The public import surface — every builder, config and class, exported in one place |
| [builder/factory.py](builder/factory.py) | The builders themselves, and the registries mapping YAML names to classes |
| [cli.py](cli.py) | The `agentic-ai` command: a thin command line over those builders |
| [utils/configs.py](utils/configs.py) | Reads the YAML into the `*Configs` dataclasses everything else is built from |
| [builder/agents.py](builder/agents.py) | The agent classes — worker, judger, classifier, planner, RAG builder |
| [builder/prompts.py](builder/prompts.py) | Reads an agent's `.md` prompts and fills in their `{arguments}` |
| [builder/tools.py](builder/tools.py) | Imports tool functions and renders their schemas per provider |
| [models/llms.py](models/llms.py) | One caller per LLM provider, behind a shared `_call` |
| [models/embeddings.py](models/embeddings.py) | One caller per embeddings provider |
| [db_connector/vector.py](db_connector/vector.py) | Vector db connectors, behind a shared `upsert/query/delete/count` |
| [db_connector/text.py](db_connector/text.py) | Text db connectors — the id-to-document half of retrieval |
| [db_connector/tabular.py](db_connector/tabular.py) | SQL db connectors |
| [benchmarks/](benchmarks/) | A worked example config |

---

## Status

Working: the config layer, the LLM and embeddings callers, the vector/text/SQL connectors, the tool and
prompt builders, the agent classes, the `agent_builder` import surface over all of it, and the
`generate agent`, `generate llm_caller` and `generate vector_db` commands.

Not implemented yet:

- **`generate agentic_ai`** — the whole-pipeline build. Its command and flags exist; the body that walks
  the `pipeline:` block (honouring `loops`, `next_step` and `needs_judger`, and feeding each agent's
  output into the next) is still to be written.
- **`orchestrators:`** — declared in the config and consumed by the pipeline runner above, not yet by
  anything else.
- **`mcp_servers:`** — an agent's `mcp_servers` names are passed straight through to the provider call;
  resolving them against a top-level `mcp_servers:` block is still to be wired up.

## License

MIT — see [LICENSE](LICENSE).
