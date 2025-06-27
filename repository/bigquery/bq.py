import pandas as pd
from .client import Client


class BQ(Client):
    def __init__(self):
        super().__init__()

    def get_sql_data(self, query):
        return self.get(query)

    def delete_text_by_id(self, text_id: str):
        query = f'''
            DELETE FROM {self.rag_table}
            WHERE text_id = "{text_id}"
        '''
        return self.execute_dml_operation(query)

    def get_text_by_id(self, task_id: str, text_id: str):
        query = f'''
            SELECT text_id 
            FROM {self.rag_table}
            WHERE text_id = "{text_id}"
                AND task_id = "{task_id}"
        '''
        return self.get(query)

    def get_nearest_text(self, vector: list[float], task_id: str, topn: int = 5) -> pd.DataFrame:
        vector_str = ','.join([str(x) for x in vector])
        sql = f'''
            WITH query_vec AS (
                SELECT [{vector_str}] AS embedding)

            SELECT base.content AS content
            FROM VECTOR_SEARCH(
                (
                SELECT content, embedding
                FROM  {self.rag_table}
                WHERE task_id = "{task_id}"),
                'embedding',
                TABLE query_vec,
                top_k => {topn})
                '''
        return self.get(sql)
