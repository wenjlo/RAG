import re
from typing import Optional

import pandas as pd
import pandas_gbq


class Client:
    _INSERT_TABLE_PATTERN = r'^[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+$'

    def __init__(self, execute_project_id: str, credentials: Optional[any] = None):
        self._execute_project_id = execute_project_id
        self._credentials = credentials

    def insert(self, table: str, df: pd.DataFrame):
        """
        Inserts data from a Pandas DataFrame into a specified BigQuery table.

        Args:
        - table (str): The fully qualified name of the destination BigQuery table in the format
          'project.dataset.table'.
        - df (pd.DataFrame): The DataFrame containing the data to be inserted.

        Returns:
        - None
        """
        if not self.is_valid_bigquery_table(table):
            raise Exception(
                f"invalid bigquery table format, insert table: {table}, required format: [project.dataset.table]")
        pandas_gbq.to_gbq(df, credentials=self._credentials, project_id=self._execute_project_id,
                          destination_table=table, if_exists='append')

    def get(self, query: str) -> pd.DataFrame:
        """
        Executes a SQL query against BigQuery and returns the results as a Pandas DataFrame.

        Args:
        - query (str): The SQL query to be executed against BigQuery.

        Returns:
        - pd.DataFrame: A DataFrame containing the query results.
        """
        return pandas_gbq.read_gbq(query, project_id=self._execute_project_id, credentials=self._credentials,
                                   use_bqstorage_api=True)

    def execute_dml_operation(self, operation: str):
        """
        Executes a Data Manipulation Language (DML) operation against BigQuery.

        Args:
        - operation (str): The DML operation to execute. Supported operations include 'delete' and 'insert_select'.

        Raises:
        - ValueError: If the provided 'operation' is not supported.

        Returns:
        - None
        """
        operation = operation.strip() + ";" if not operation.strip().endswith(";") else operation.strip()
        operation = f'{operation} select 1;'
        pandas_gbq.read_gbq(operation, project_id=self._execute_project_id, credentials=self._credentials)

    @staticmethod
    def is_valid_bigquery_table(table: str) -> bool:
        """
        Check if the given table is a valid BigQuery table identifier in the format project.dataset.table.

        Args:
            table (str): The BigQuery table identifier to check.

        Returns:
            bool: True if the table is valid, False otherwise.
        """
        return bool(re.match(Client._INSERT_TABLE_PATTERN, table))
