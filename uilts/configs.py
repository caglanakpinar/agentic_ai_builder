

from logging import config
from pathlib import Path
import yaml


class VectorDBConfigs:
    def __init__(self, host: str, port: int, api_key: str, path: str):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.path = path

class Configs:
    db_confgs: dict[str, VectorDBConfigs] = {}
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
 