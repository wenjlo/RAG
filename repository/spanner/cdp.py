from config import config
from .client import Client


class Spanner(Client):
    _DATABASE = config.spanner_cdp_database

    def __init__(self):
        super().__init__(database_id=self._DATABASE)

    def get_sql_data(self, query):
        return self.get(query)


