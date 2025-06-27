from config import config
from submodule.spanner import spanner


class Client(spanner.Client):
    _PROJECT = config.spanner_project
    _INSTANCE = config.spanner_instance

    def __init__(self, database_id: str):
        super().__init__(self._PROJECT, self._INSTANCE, database_id)
