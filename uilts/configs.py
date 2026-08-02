from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class LLMConfigs:
    model_name: str
    temperature: float
    max_tokens: int
    api_key: str
    tools: list[dict[str, str]] | None = None
    type: str = "generator" # options are "generator" or "retriever", "tool caller", "tool generator"
    mcp_servers: list[str] | None = None


@dataclass
class AgentConfigs:
    responsiblity_prompt: str
    tools: list[dict[str, str]] | None = None
    type: str = "generator" # options are "generator" or "retriever", "tool caller", "tool generator"
    mcp_servers: list[str] | None = None
    llms: dict[str, LLMConfigs] | None = None
    db: str | None = None  # name of the vector/knowledge db this agent retrieves from, e.g. "ds_knowledge_db"
    prompt: str | None = None  # directory (relative to Configs.current_dir) containing one file per prompt, e.g. "prompts/rag_problem_thinker_agent"
    arguments: list[str] = field(default_factory=list)  # prompt names discovered in `prompt`, filled in by BasePrompt.prompt_configer

@dataclass
class VectorDBConfigs:
    name: str  # db name as referenced by agents, e.g. "ds_knowledge_db"
    type: str  # driver to connect with: "faiss", "chroma", "qdrant", "pinecone", "weaviate", ...
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
    type: str  # driver to connect with: "postgresql", "mysql", "bigquery", "sqlite", "snowflake", ...
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
    db_confgs: dict[str, VectorDBConfigs] = {}
    llm_configs: dict[str, LLMConfigs] = {}
    embeddings_configs: dict[str, EmbeddingsConfigs] = {}
    agent_configs: dict[str, AgentConfigs] = {}

    def __init__(self, current_filename: str):
        self.current_dir = Path(__file__).parent  / current_filename
        self.read_yaml()
        self.db_configer()
        self.llm_configer()
        self.embeddings_configer()
        self.agent_configer()

    def read_yaml(self):
        for file in self.current_dir.glob("*.yaml"):
            with open(file, 'r') as f:
                configs = yaml.safe_load(f) or {}

            for key, value in configs.items():
                setattr(self, key, value)
            return

        raise FileNotFoundError(f"No YAML configuration file found in {self.current_dir}.")

    def db_configer(self):
        if 'dbs' in self.__dict__:
            for db_name, db_cfg in self.dbs.items():
                if db_name == 'vector_db':
                    self.db_confgs[db_name] = VectorDBConfigs(
                        host=db_cfg.get('host', 'localhost'),
                        port=db_cfg.get('port', 5432),
                        api_key=db_cfg.get('api_key', ''),
                        path=db_cfg.get('path', '')
                    )

    def llm_configer(self):
        if 'llm' in self.__dict__:
            for llm_name, llm_cfg in self.llm.items():
                self.llm_configs[llm_name] = LLMConfigs(
                    model_name=llm_cfg.get('model_name', ''),
                    temperature=llm_cfg.get('temperature', 0.0),
                    max_tokens=llm_cfg.get('max_tokens', 0),
                    api_key=llm_cfg.get('api_key', ''),
                    type=llm_cfg.get('type', 'generator'),
                    tools=llm_cfg.get('tools', None),
                    mcp_servers=llm_cfg.get('mcp_servers', None)
                )

    def embeddings_configer(self):
        if 'embeddings' in self.__dict__:
            for embedding_name, embedding_cfg in self.embeddings.items():
                self.embeddings_configs[embedding_name] = EmbeddingsConfigs(
                    model_name=embedding_cfg.get('model_name', ''),
                    api_key=embedding_cfg.get('api_key', '')
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
                    db=agent_cfg.get('db', None),
                    prompt=agent_cfg.get('prompt', None)
                )