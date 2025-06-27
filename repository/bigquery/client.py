from config import config
from submodule.bigquery import bigquery


class Client(bigquery.Client):
    _EXECUTE_PROJECT = config.etl_gcp_project

    _CDP_PROJECT = config.bigquery_project
    _CDP_RAG_DATASET = 'rag_dataset'
    _rag_table = 'rag_table'

    def __init__(self):
        super().__init__(execute_project_id=self._EXECUTE_PROJECT)

    @property
    def rag_dataset(self) -> str:
        return f'{self._CDP_PROJECT}.{self._CDP_RAG_DATASET}'

    @property
    def rag_table(self) -> str:
        return f'{self.rag_dataset}.{self._rag_table}'