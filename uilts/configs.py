from dataclasses import dataclass
from logging import config
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
class VectorDBConfigs:
    host: str
    port: int
    api_key: str
    path: str


@dataclass
class EmbeddingsConfigs:
    model_name: str
    api_key: str


class Configs:
    db_confgs: dict[str, VectorDBConfigs] = {}
    llm_configs: dict[str, LLMConfigs] = {}
    embeddings_configs: dict[str, EmbeddingsConfigs] = {}

    def __init__(self, current_filename: str):
        self.current_dir = Path(__file__).parent  / current_filename
        self.read_yaml()

    def read_yaml(self): 
        for file in self.current_dir.listdir():
            if file.is_file() and file.suffix == '.yaml':
                with open(file, 'r') as f:
                    configs = yaml.safe_load(f)

                for key, value in config.items():
                    if key in self._YAML_KEYS:
                        setattr(self, key, value)
                return 

        raise FileNotFoundError("No YAML configuration file found in the current directory.")

    def db_configer(self):
        if 'dbs' in self.__dict__:
            for db_name in self.dbs:
                if db_name == 'vector_db':
                    self.db_confgs[db_name] = VectorDBConfigs(
                        host=self.dbs.get('host', 'localhost'),
                        port=self.dbs.get('port', 5432),
                        api_key=self.dbs.get('api_key', ''),
                        path=self.dbs.get('path', '')
                    )

    def llm_configer(self):
        if 'llm' in self.__dict__:
            for llm_name in self.llm:
                self.llm_configs[llm_name] = LLMConfigs(
                    model_name=self.llm.get('model_name', ''),
                    temperature=self.llm.get('temperature', 0.0),
                    max_tokens=self.llm.get('max_tokens', 0),
                    api_key=self.llm.get('api_key', ''),
                    type=self.llm.get('type', 'generator'),
                    tools=self.llm.get('tools', None)
                )

    def embeddings_configer(self):
        if 'embeddings' in self.__dict__:
            for embedding_name in self.embeddings:
                self.embeddings_configs[embedding_name] = EmbeddingsConfigs(
                    model_name=self.embeddings.get('model_name', ''),
                    api_key=self.embeddings.get('api_key', '')
                )
