import concurrent.futures
import logging
import sys
import time
import uuid

import pandas as pd
from google.api_core import exceptions
from google.api_core.retry import Retry
from google.cloud import spanner
from grpc import StatusCode


class Client:
    # 讀取操作重試參數
    _READ_RETRY_INITIAL = 0.1  # 初始重試延遲 (秒)
    _READ_RETRY_MAXIMUM = 5.0  # 最大延遲時間 (秒)
    _READ_RETRY_MULTIPLIER = 2.0  # 延遲倍增係數
    _READ_RETRY_DEADLINE = 30.0  # 最大重試時間 (秒)

    # 寫入操作重試參數
    _WRITE_RETRY_INITIAL = 0.2  # 初始重試延遲 (秒)
    _WRITE_RETRY_MAXIMUM = 30.0  # 最大延遲時間 (秒)
    _WRITE_RETRY_MULTIPLIER = 1.5  # 延遲倍增係數
    _WRITE_RETRY_DEADLINE = 120.0  # 最大重試時間 (秒)

    # 可重試的 gRPC 狀態碼
    _RETRYABLE_STATUS_CODES = {
        StatusCode.UNAVAILABLE,  # 服務暫時不可用
        StatusCode.DEADLINE_EXCEEDED,  # 操作超時
        StatusCode.RESOURCE_EXHAUSTED,  # 資源用盡（配額超出或速率限制）
        StatusCode.ABORTED,  # 事務中止（可重試）
        StatusCode.INTERNAL,  # 服務內部錯誤（可能是暫時的）
    }
    _RETRYABLE_HTTP_CODES = {
        408,  # Request Timeout - 伺服器等待請求時間過長而終止連接
        429,  # Too Many Requests - 因為請求速率限制而被拒絕（需要限流）
        500,  # Internal Server Error - 伺服器遇到了未預期的錯誤
        502,  # Bad Gateway - 上游伺服器收到無效回應
        503,  # Service Unavailable - 伺服器暫時無法處理請求（過載或維護）
        504  # Gateway Timeout - 伺服器作為閘道等待上游回應超時
    }

    _POOL_SIZE = 200
    _MUTATION_LIMIT = 80000

    # 直接引用官方 param_types, COMMIT_TIMESTAMP
    param_types = spanner.param_types
    COMMIT_TIMESTAMP = spanner.COMMIT_TIMESTAMP

    def __init__(self, project_id: str, instance_id: str, database_id: str, threads: int = 5):
        """
        Initialize the SpannerClient with project, instance, and database IDs.

        Args:
            project_id: Google Cloud project ID
            instance_id: Spanner instance ID
            database_id: Spanner database ID
        """
        self.client = spanner.Client(project=project_id)
        self.instance = self.client.instance(instance_id)
        self.database = self.instance.database(
            database_id,
            pool=spanner.BurstyPool(target_size=self._POOL_SIZE),
        )
        self.database.log_commit_stats = True
        self.threads = threads

        self.read_retry = Retry(
            initial=self._READ_RETRY_INITIAL,
            maximum=self._READ_RETRY_MAXIMUM,
            multiplier=self._READ_RETRY_MULTIPLIER,
            deadline=self._READ_RETRY_DEADLINE,
        )
        self.write_retry = Retry(
            initial=self._WRITE_RETRY_INITIAL,
            maximum=self._WRITE_RETRY_MAXIMUM,
            multiplier=self._WRITE_RETRY_MULTIPLIER,
            deadline=self._WRITE_RETRY_DEADLINE,
        )

        self._setup_logger()
        self.logger.info(f"Initialized Spanner Client for {project_id}/{instance_id}/{database_id}")

    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _is_retryable_exception(self, exc: Exception) -> bool:
        """
        Determine if an exception should be retried.

        Args:
            exc: The exception to check

        Returns:
            bool: True if the exception should be retried, False otherwise
        """
        # 處理 Google API 調用錯誤
        if isinstance(exc, exceptions.GoogleAPICallError):
            grpc_code = getattr(exc, 'grpc_status_code', None)
            if grpc_code in self._RETRYABLE_STATUS_CODES:
                return True

        # 處理連接相關錯誤
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
            return True

        # 處理 Spanner 特定的錯誤 (如果適用)
        if hasattr(exc, 'code'):
            # 會話過期錯誤
            if str(exc.code) == 'FAILED_PRECONDITION' or 'Session not found' in str(exc):
                return True
            # 事務衝突
            if str(exc.code) == 'ABORTED':
                return True

        # 處理特定的 HTTP 錯誤（如果使用 REST API）
        if hasattr(exc, 'status_code'):
            if exc.status_code in self._RETRYABLE_HTTP_CODES:
                return True

        # 默認不重試
        return False

    def _retry_write_operation(self, func, *args, **kwargs):
        """
        A function to retry an operation with exponential backoff.

        Args:
            func: The function to retry
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        delay = self._WRITE_RETRY_INITIAL
        deadline = time.time() + self._WRITE_RETRY_DEADLINE
        attempt = 0

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 檢查是否應該重試此異常
                if not self._is_retryable_exception(e):
                    raise e

                # 檢查是否超過截止時間
                now = time.time()
                remaining = deadline - now
                if remaining <= 0:
                    self.logger.error("Retry deadline exceeded, operation failed.")
                    raise e

                attempt += 1

                # 計算下一次重試延遲，需同時考慮三個限制:
                # 1. 應用退避乘數後的延遲
                # 2. 最大延遲限制
                # 3. 剩餘到截止時間的時間
                delay = min(
                    delay * self._WRITE_RETRY_MULTIPLIER,  # 應用退避乘數
                    self._WRITE_RETRY_MAXIMUM,  # 不超過最大延遲
                    remaining  # 不超過剩餘時間
                )

                self.logger.warning(f"Retry attempt {attempt} in {delay:.2f}s...")
                time.sleep(delay)

    @staticmethod
    def _chunk_dataframe(df: pd.DataFrame, max_rows: int) -> list[pd.DataFrame]:
        return [df.iloc[i:i + max_rows] for i in range(0, len(df), max_rows)]

    def _process_batch(self, table: str, chunk_df: pd.DataFrame, operation: str):
        values = chunk_df.values.tolist()
        columns = chunk_df.columns.tolist()

        with self.database.batch() as batch:
            if operation == 'insert':
                self._retry_write_operation(batch.insert, table=table, columns=columns, values=values)
            elif operation == 'upsert':
                self._retry_write_operation(batch.insert_or_update, table=table, columns=columns, values=values)
            elif operation == 'update':
                self._retry_write_operation(batch.update, table=table, columns=columns, values=values)
            elif operation == 'delete':
                key_set = spanner.KeySet(keys=values)
                self._retry_write_operation(batch.delete, table, key_set)
            else:
                raise ValueError(f"Operation {operation} not supported")

        commit_stats = batch.commit_stats

        self.logger.info(
            f"Successfully processed {len(chunk_df)} rows in {table} using {operation} operation, "
            f"commit_stats: {commit_stats}")

    def _execute_in_threads(self, table: str, df: pd.DataFrame, operation: str):
        column_count = df.shape[1]
        mutation_per_row = self._get_mutation_per_row(table, column_count)
        max_rows_per_batch = self._MUTATION_LIMIT // mutation_per_row

        chunks = self._chunk_dataframe(df, max_rows_per_batch)
        total_chunks = len(chunks)

        if total_chunks == 1:
            self._process_batch(table, chunks[0], operation)
        else:
            worker_count = min(self.threads, total_chunks)
            self.logger.info(f"啟動 {worker_count} 個線程處理 {total_chunks} 個批次")

            completed = 0
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(self._process_batch, table, chunk_df, operation): i
                           for i, chunk_df in enumerate(chunks)}

                for future in concurrent.futures.as_completed(futures):
                    completed += 1
                    chunk_idx = futures[future]
                    try:
                        future.result()
                        self.logger.info(
                            f"進度: {completed}/{total_chunks} 批次 ({completed / total_chunks * 100:.1f}%)")
                    except Exception as e:
                        error_msg = f"批次 {chunk_idx + 1}/{total_chunks} 處理失敗: {str(e)}"
                        self.logger.error(error_msg)
                        errors.append(e)

            if errors:
                raise Exception(f"執行中發生 {len(errors)} 個錯誤: {errors}")

    def get(self, query: str, params: dict[str, any] = None,
            param_types: dict[str, spanner.param_types] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a pandas DataFrame.

        Args:
            query: SQL query string to execute
            params: Dictionary of parameter values
            param_types: Dictionary of Spanner param types (e.g., STRING, INT64, etc.)

        Returns:
            pandas DataFrame containing query results
        """
        self.logger.debug(f"Executing query: {query}")

        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                sql=query,
                params=params or {},
                param_types=param_types or {},
                retry=self.read_retry,
            )

            rows = list(results)
            columns = [field.name for field in results.fields]
            df = pd.DataFrame(rows, columns=columns)
            self.logger.info(f"Query returned {len(df)} rows")

            return df

    def insert(self, table: str, insert_df: pd.DataFrame):
        """
        Insert new rows into a table.

        Args:
            table: Table name to insert into
            insert_df: DataFrame containing rows to insert
        """
        if insert_df.empty:
            self.logger.warning(f"No data to insert into {table}")
            return

        self.logger.info(f"Inserting {len(insert_df)} rows into {table}")

        self._execute_in_threads(table, insert_df, operation='insert')

    def upsert(self, table: str, upsert_df: pd.DataFrame):
        """
        Insert or update rows in a table.

        Args:
            table: Table name to upsert into
            upsert_df: DataFrame containing rows to upsert
        """
        if upsert_df.empty:
            self.logger.warning(f"No data to upsert into {table}")
            return

        self.logger.info(f"Upserting {len(upsert_df)} rows into {table}")

        self._execute_in_threads(table, upsert_df, operation='upsert')

    def update(self, table: str, update_df: pd.DataFrame):
        """
        Update rows in a table using mutation API.

        Args:
            table: Table name to update
            update_df: DataFrame containing rows to update with primary keys
        """
        if update_df.empty:
            self.logger.warning(f"No data to update in {table}")
            return

        self.logger.info(f"Updating {len(update_df)} rows in {table}")

        self._execute_in_threads(table, update_df, operation='update')

    def delete_by_keys(self, table: str, primary_keys: list[str], delete_df: pd.DataFrame):
        """
        Delete rows from a table based on primary keys.

        Args:
            table: Table name to delete from
            primary_keys: List of primary key column names
            df: DataFrame containing primary key values for rows to delete
        """
        if delete_df.empty:
            self.logger.warning(f"No data to delete from {table}")
            return

        self.logger.info(f"Deleting {len(delete_df)} rows from {table}")

        delete_df = delete_df[primary_keys]
        self._execute_in_threads(table, delete_df, operation='delete')

    def execute_statement(self, statement: str, params: dict[str, any] = None,
                          param_types: dict[str, spanner.param_types] = None) -> pd.DataFrame:
        """
        Execute a statement with THEN RETURN clause and return the result as a pandas DataFrame.

        Args:
            statement: SQL statement with THEN RETURN clause.
            params: Dictionary of parameter values.
            param_types: Dictionary of Spanner param types (e.g., STRING, INT64, etc.)

        Returns:
            A pandas DataFrame containing the rows returned by THEN RETURN.
        """
        self.logger.debug(
            f"Executing statement with THEN RETURN: {statement} with params: {params} and param_types: {param_types}")

        def txn_fn(transaction) -> pd.DataFrame:
            results = transaction.execute_sql(
                statement,
                params=params or {},
                param_types=param_types or {},
                retry=self.write_retry,
            )
            rows = list(results)
            columns = [field.name for field in results.fields]
            return pd.DataFrame(rows, columns=columns)

        df = self.database.run_in_transaction(txn_fn)
        self.logger.info(f"Statement executed with THEN RETURN, {len(df)} rows returned.")
        return df

    def execute_partitioned_dml(self, statement: str, params: dict[str, any] = None,
                                param_types: dict[str, spanner.param_types] = None):
        """
        Execute a partitioned DML statement, suitable for large-scale update or delete operations.

        Args:
            statement: SQL DML statement to execute
            params: Dictionary of parameters (optional)
            param_types: Dictionary of parameter types (optional)

        Returns:
            row_count: Number of rows affected by the operation
        """
        operation_id = str(uuid.uuid4())[:8]
        self.logger.info(f"[PartitionedDML:{operation_id}] Executing statement: {statement}")
        start_time = time.time()

        try:
            row_count = self._retry_write_operation(
                self.database.execute_partitioned_dml,
                statement,
                params=params or {},
                param_types=param_types or {},
            )

            elapsed_time = time.time() - start_time
            self.logger.info(
                f"[PartitionedDML:{operation_id}] Statement executed successfully in {elapsed_time:.2f} seconds. "
                f"Affected rows: {row_count}")

            return row_count
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(
                f"[PartitionedDML:{operation_id}] Failed after {elapsed_time:.2f} seconds. Error: {str(e)}")
            raise

    def _get_mutation_per_row(self, table: str, column_count: int) -> int:
        query = f"""
            SELECT 
                `INDEX_NAME`,
                `COLUMN_NAME`,
                `ORDINAL_POSITION`
            FROM information_schema.index_columns
            WHERE `TABLE_NAME` = @table
                AND `INDEX_TYPE` IN ('INDEX', 'PRIMARY_KEY')
        """
        params = {
            'table': table,
        }
        param_types = {
            'table': self.param_types.STRING
        }
        df = self.get(query, params, param_types)

        is_pk = df['INDEX_NAME'] == 'PRIMARY_KEY'
        is_idx = ~is_pk

        primary_key_count = df.loc[is_pk, 'COLUMN_NAME'].nunique()
        secondary_index_count = df.loc[is_idx, 'INDEX_NAME'].nunique()
        storing_columns_count = df.loc[is_idx & df['ORDINAL_POSITION'].isna()].shape[0]

        return column_count + primary_key_count + secondary_index_count + storing_columns_count
