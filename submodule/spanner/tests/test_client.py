from unittest import mock

import pandas as pd
import pytest
from google.api_core import exceptions
from google.cloud import spanner
from grpc import StatusCode

from spanner import Client
from tests.conftest import MockSpannerClient, MockResultSet


class TestClient:
    def test_init(self, mock_spanner_client):
        """測試初始化 Client 的實例"""
        client = Client("test-project", "test-instance", "test-database")
        assert client.threads == 5
        mock_spanner_client.assert_called_once_with(project="test-project")

    def test_setup_logger(self):
        """測試日誌設置"""
        with mock.patch('google.cloud.spanner.Client', return_value=MockSpannerClient()), \
                mock.patch('google.cloud.spanner.BurstyPool', return_value=None), \
                mock.patch('logging.getLogger') as mock_get_logger, \
                mock.patch('logging.StreamHandler') as mock_handler:
            mock_logger = mock.MagicMock()
            mock_get_logger.return_value = mock_logger
            mock_logger.handlers = []

            client = Client("test-project", "test-instance", "test-database")

            mock_get_logger.assert_called_once()
            assert mock_logger.setLevel.called
            assert mock_handler.called

    def test_is_retryable_exception(self, client):
        """測試異常是否可重試的判斷邏輯"""
        # 測試 GoogleAPICallError 異常
        google_api_error = mock.MagicMock(spec=exceptions.GoogleAPICallError)
        google_api_error.grpc_status_code = StatusCode.UNAVAILABLE
        assert client._is_retryable_exception(google_api_error) is True

        # 測試一般連接錯誤
        connection_error = ConnectionError("Connection refused")
        assert client._is_retryable_exception(connection_error) is True

        # 測試非重試異常
        value_error = ValueError("Invalid argument")
        assert client._is_retryable_exception(value_error) is False

        # 測試 ABORTED 錯誤碼
        aborted_error = mock.MagicMock()
        aborted_error.code = "ABORTED"
        assert client._is_retryable_exception(aborted_error) is True

        # 測試會話過期錯誤
        session_error = mock.MagicMock()
        session_error.code = "FAILED_PRECONDITION"
        assert client._is_retryable_exception(session_error) is True

        # 測試 HTTP 錯誤
        http_error = mock.MagicMock()
        http_error.status_code = 503
        assert client._is_retryable_exception(http_error) is True

        # 測試非重試 HTTP 錯誤
        http_error.status_code = 400
        assert client._is_retryable_exception(http_error) is False

    def test_retry_write_operation_success(self, client):
        """測試重試操作成功的情況"""
        mock_func = mock.MagicMock()
        mock_func.return_value = "success"

        result = client._retry_write_operation(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_retry_write_operation_retries_and_succeeds(self, client):
        """測試重試操作在失敗後重試並最終成功的情況"""
        mock_func = mock.MagicMock()

        # 創建一個真實的可重試異常
        retryable_error = exceptions.GoogleAPICallError("Temporary unavailable")
        # 為異常添加 grpc_status_code 屬性
        retryable_error.grpc_status_code = StatusCode.UNAVAILABLE

        # 設置 side_effect，前兩次調用拋出異常，第三次返回成功
        mock_func.side_effect = [retryable_error, retryable_error, "success"]

        with mock.patch('time.sleep') as mock_sleep, \
                mock.patch('time.time', side_effect=[0, 1, 2, 3, 4]):
            result = client._retry_write_operation(mock_func, "arg1")

            assert result == "success"
            assert mock_func.call_count == 3
            assert mock_sleep.call_count == 2

    def test_retry_write_operation_non_retryable_error(self, client):
        """測試遇到不可重試的錯誤時不重試的情況"""
        mock_func = mock.MagicMock()
        non_retryable_error = ValueError("Invalid value")
        mock_func.side_effect = non_retryable_error

        with pytest.raises(ValueError):
            client._retry_write_operation(mock_func, "arg1")

        mock_func.assert_called_once()

    def test_retry_write_operation_deadline_exceeded(self, client):
        """測試重試超過截止時間的情況"""
        mock_func = mock.MagicMock()

        # 創建一個真實的可重試異常
        retryable_error = exceptions.GoogleAPICallError("Temporary unavailable")
        # 為異常添加 grpc_status_code 屬性
        retryable_error.grpc_status_code = StatusCode.UNAVAILABLE

        mock_func.side_effect = retryable_error

        # 模擬時間快速前進超過截止時間
        with mock.patch('time.sleep') as mock_sleep, \
                mock.patch('time.time', side_effect=[0, client._WRITE_RETRY_DEADLINE + 1]):
            with pytest.raises(exceptions.GoogleAPICallError):
                client._retry_write_operation(mock_func, "arg1")

            mock_func.assert_called_once()
            mock_sleep.assert_not_called()

    def test_chunk_dataframe(self, client):
        """測試 DataFrame 分塊功能"""
        # 創建一個有 15 行的測試 DataFrame
        df = pd.DataFrame({'A': range(15), 'B': range(15, 30)})

        # 測試以最大 5 行進行分塊
        chunks = client._chunk_dataframe(df, 5)

        assert len(chunks) == 3
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 5
        assert len(chunks[2]) == 5
        assert chunks[0].iloc[0, 0] == 0
        assert chunks[1].iloc[0, 0] == 5
        assert chunks[2].iloc[0, 0] == 10

    def test_process_batch_insert(self, client):
        """測試批處理插入操作"""
        # 創建測試數據
        df = pd.DataFrame({'id': [1, 2], 'name': ['test1', 'test2']})

        # 模擬批處理
        mock_batch = mock.MagicMock()
        mock_batch.__enter__.return_value = mock_batch
        mock_batch.__exit__.return_value = None

        with mock.patch.object(client.database, 'batch', return_value=mock_batch):
            client._process_batch('test_table', df, 'insert')

            mock_batch.insert.assert_called_once_with(
                table='test_table',
                columns=['id', 'name'],
                values=[[1, 'test1'], [2, 'test2']]
            )
            assert mock_batch.__exit__.called

    def test_process_batch_upsert(self, client):
        """測試批處理 upsert 操作"""
        df = pd.DataFrame({'id': [1, 2], 'name': ['test1', 'test2']})

        mock_batch = mock.MagicMock()
        mock_batch.__enter__.return_value = mock_batch
        mock_batch.__exit__.return_value = None

        with mock.patch.object(client.database, 'batch', return_value=mock_batch):
            client._process_batch('test_table', df, 'upsert')

            mock_batch.insert_or_update.assert_called_once_with(
                table='test_table',
                columns=['id', 'name'],
                values=[[1, 'test1'], [2, 'test2']]
            )
            assert mock_batch.__exit__.called

    def test_process_batch_update(self, client):
        """測試批處理更新操作"""
        df = pd.DataFrame({'id': [1, 2], 'name': ['test1', 'test2']})

        mock_batch = mock.MagicMock()
        mock_batch.__enter__.return_value = mock_batch
        mock_batch.__exit__.return_value = None

        with mock.patch.object(client.database, 'batch', return_value=mock_batch):
            client._process_batch('test_table', df, 'update')

            mock_batch.update.assert_called_once_with(
                table='test_table',
                columns=['id', 'name'],
                values=[[1, 'test1'], [2, 'test2']]
            )
            assert mock_batch.__exit__.called

    def test_process_batch_invalid_operation(self, client):
        """測試無效的批處理操作"""
        df = pd.DataFrame({'id': [1, 2], 'name': ['test1', 'test2']})

        with pytest.raises(ValueError, match="Operation invalid_op not supported"):
            client._process_batch('test_table', df, 'invalid_op')

    def test_execute_in_threads(self, client):
        """測試多線程執行操作"""
        # 創建測試數據
        df = pd.DataFrame({'id': range(50000), 'name': [f'test{i}' for i in range(50000)]})

        # 模擬 _process_batch 方法
        with mock.patch.object(client, '_process_batch') as mock_process_batch, \
                mock.patch.object(client, 'get') as mock_get:
            mock_get.return_value = pd.DataFrame({
                'INDEX_NAME': ['PRIMARY_KEY', 'index1', 'index1', 'index1'],
                'COLUMN_NAME': ['column1', 'column2', 'column3', 'column1'],
                'ORDINAL_POSITION': [1, 2, 3, pd.NA],
            })

            client._execute_in_threads('test_table', df, 'insert')

            # 應該調用 4 次 (10 行數據，每 3 行一批)
            assert mock_process_batch.call_count == 4

            # 檢查每次調用的參數
            calls = mock_process_batch.call_args_list
            assert len(calls[0][0][1]) == 16000  # 第一批 16000 行
            assert len(calls[1][0][1]) == 16000  # 第二批 16000 行
            assert len(calls[2][0][1]) == 16000  # 第三批 16000 行
            assert len(calls[3][0][1]) == 2000  # 第四批 2000 行

    def test_get(self, client):
        """測試 get 方法執行查詢並返回 DataFrame"""
        # 模擬查詢結果
        mock_results = MockResultSet(
            rows=[(1, 'test1'), (2, 'test2')],
            field_names=['id', 'name']
        )

        # 模擬快照和執行 SQL
        mock_snapshot = mock.MagicMock()
        mock_snapshot.__enter__.return_value.execute_sql.return_value = mock_results

        with mock.patch.object(client.database, 'snapshot', return_value=mock_snapshot):
            # 執行測試
            result_df = client.get("SELECT * FROM test_table")

            # 驗證
            mock_snapshot.__enter__.return_value.execute_sql.assert_called_once()
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 2
            assert list(result_df.columns) == ['id', 'name']
            assert result_df.iloc[0, 0] == 1
            assert result_df.iloc[0, 1] == 'test1'

    def test_get_with_params(self, client):
        """測試帶參數的 get 方法"""
        # 模擬查詢結果
        mock_results = MockResultSet(
            rows=[(1, 'test1')],
            field_names=['id', 'name']
        )

        # 模擬快照和執行 SQL
        mock_snapshot = mock.MagicMock()
        mock_snapshot.__enter__.return_value.execute_sql.return_value = mock_results

        with mock.patch.object(client.database, 'snapshot', return_value=mock_snapshot):
            # 執行測試
            params = {'id': 1}
            param_types = {'id': spanner.param_types.INT64}
            result_df = client.get(
                "SELECT * FROM test_table WHERE id = @id",
                params=params,
                param_types=param_types
            )

            # 驗證
            mock_snapshot.__enter__.return_value.execute_sql.assert_called_once_with(
                sql="SELECT * FROM test_table WHERE id = @id",
                params=params,
                param_types=param_types,
                retry=client.read_retry
            )
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 1

    def test_insert(self, client):
        """測試插入操作"""
        # 創建測試數據
        df = pd.DataFrame({'id': [1, 2], 'name': ['test1', 'test2']})

        # 模擬 _execute_in_threads 方法
        with mock.patch.object(client, '_execute_in_threads') as mock_execute_in_threads:
            client.insert('test_table', df)

            mock_execute_in_threads.assert_called_once_with('test_table', df, operation='insert')

    def test_insert_empty_dataframe(self, client):
        """測試插入空 DataFrame 的情況"""
        df = pd.DataFrame()

        client.insert('test_table', df)

        # 驗證記錄了警告但沒有進行實際操作
        client.logger.warning.assert_called_once()

    def test_upsert(self, client):
        """測試 upsert 操作"""
        # 創建測試數據
        df = pd.DataFrame({'id': [1, 2], 'name': ['test1', 'test2']})

        # 模擬 _execute_in_threads 方法
        with mock.patch.object(client, '_execute_in_threads') as mock_execute_in_threads:
            client.upsert('test_table', df)

            mock_execute_in_threads.assert_called_once_with('test_table', df, operation='upsert')

    def test_upsert_empty_dataframe(self, client):
        """測試 upsert 空 DataFrame 的情況"""
        df = pd.DataFrame()

        client.upsert('test_table', df)

        # 驗證記錄了警告但沒有進行實際操作
        client.logger.warning.assert_called_once()

    def test_update(self, client):
        """測試更新操作"""
        # 創建測試數據
        df = pd.DataFrame({'id': [1, 2], 'name': ['updated1', 'updated2']})

        # 模擬 _execute_in_threads 方法
        with mock.patch.object(client, '_execute_in_threads') as mock_execute_in_threads:
            client.update('test_table', df)

            mock_execute_in_threads.assert_called_once_with('test_table', df, operation='update')

    def test_update_empty_dataframe(self, client):
        """測試更新空 DataFrame 的情況"""
        df = pd.DataFrame()

        client.update('test_table', df)

        # 驗證記錄了警告但沒有進行實際操作
        client.logger.warning.assert_called_once()

    def test_delete_by_keys(self, client):
        """測試根據主鍵刪除行"""
        # 創建測試數據
        primary_key = ['id']
        df = pd.DataFrame({'id': [1, 2], 'name': ['test1', 'test2']})

        # 模擬 _execute_in_threads 方法
        with mock.patch.object(client, '_execute_in_threads') as mock_execute_in_threads:
            client.delete_by_keys('test_table', primary_key, df)

            args, kwargs = mock_execute_in_threads.call_args

            assert args[0] == 'test_table'
            pd.testing.assert_frame_equal(args[1], df[primary_key])
            assert kwargs['operation'] == 'delete'

    def test_delete_by_keys_empty_dataframe(self, client):
        """測試刪除空 DataFrame 的情況"""
        df = pd.DataFrame()

        client.delete_by_keys('test_table', ['id'], df)

        # 驗證記錄了警告但沒有進行實際操作
        client.logger.warning.assert_called_once()

    def test_execute_statement(self, client):
        """測試執行 SQL 語句並返回結果"""
        # 模擬查詢結果
        mock_results = MockResultSet(
            rows=[(1, 'test1'), (2, 'test2')],
            field_names=['id', 'name']
        )

        # 模擬事務和執行 SQL
        mock_transaction = mock.MagicMock()
        mock_transaction.execute_sql.return_value = mock_results

        with mock.patch.object(client.database, 'run_in_transaction') as mock_run_in_txn:
            # 設置 run_in_transaction 使用提供的函數來執行並傳遞 mock_transaction
            mock_run_in_txn.side_effect = lambda fn: fn(mock_transaction)

            # 執行測試
            result_df = client.execute_statement(
                "INSERT INTO test_table (id, name) VALUES (1, 'test1') THEN RETURN id, name"
            )

            # 驗證
            mock_transaction.execute_sql.assert_called_once()
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 2
            assert list(result_df.columns) == ['id', 'name']

    def test_execute_statement_with_params(self, client):
        """測試帶參數的執行 SQL 語句"""
        # 模擬查詢結果
        mock_results = MockResultSet(
            rows=[(1, 'test1')],
            field_names=['id', 'name']
        )

        # 模擬事務和執行 SQL
        mock_transaction = mock.MagicMock()
        mock_transaction.execute_sql.return_value = mock_results

        with mock.patch.object(client.database, 'run_in_transaction') as mock_run_in_txn:
            # 設置 run_in_transaction 使用提供的函數來執行並傳遞 mock_transaction
            mock_run_in_txn.side_effect = lambda fn: fn(mock_transaction)

            # 執行測試
            params = {'id': 1, 'name': 'test1'}
            param_types = {'id': spanner.param_types.INT64, 'name': spanner.param_types.STRING}
            result_df = client.execute_statement(
                "INSERT INTO test_table (id, name) VALUES (@id, @name) THEN RETURN id, name",
                params=params,
                param_types=param_types
            )

            # 驗證
            mock_transaction.execute_sql.assert_called_once_with(
                "INSERT INTO test_table (id, name) VALUES (@id, @name) THEN RETURN id, name",
                params=params,
                param_types=param_types,
                retry=client.write_retry
            )
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 1

    def test_execute_partitioned_dml(self, client):
        """測試執行分割 DML 語句"""
        # 模擬 database.execute_partitioned_dml 方法
        with mock.patch.object(client.database, 'execute_partitioned_dml', return_value=100), \
                mock.patch('uuid.uuid4', return_value=mock.MagicMock(hex='12345678')):
            # 執行測試
            result = client.execute_partitioned_dml(
                "UPDATE test_table SET name = 'updated' WHERE id > 100"
            )

            # 驗證
            client.database.execute_partitioned_dml.assert_called_once_with(
                "UPDATE test_table SET name = 'updated' WHERE id > 100",
                params={},
                param_types={}
            )
            assert result == 100
            client.logger.info.assert_called()

    def test_execute_partitioned_dml_with_params(self, client):
        """測試帶參數的執行分割 DML 語句"""
        # 模擬 database.execute_partitioned_dml 方法
        with mock.patch.object(client.database, 'execute_partitioned_dml', return_value=50), \
                mock.patch('uuid.uuid4', return_value=mock.MagicMock(hex='12345678')):
            # 執行測試
            params = {'min_id': 100}
            param_types = {'min_id': spanner.param_types.INT64}
            result = client.execute_partitioned_dml(
                "UPDATE test_table SET name = 'updated' WHERE id > @min_id",
                params=params,
                param_types=param_types
            )

            # 驗證
            client.database.execute_partitioned_dml.assert_called_once_with(
                "UPDATE test_table SET name = 'updated' WHERE id > @min_id",
                params=params,
                param_types=param_types
            )
            assert result == 50
            client.logger.info.assert_called()

    def test_execute_partitioned_dml_error(self, client):
        """測試執行分割 DML 語句出錯的情況"""
        # 模擬 database.execute_partitioned_dml 方法拋出異常
        with mock.patch.object(client.database, 'execute_partitioned_dml',
                               side_effect=ValueError("Test error")), \
                mock.patch('uuid.uuid4', return_value=mock.MagicMock(hex='12345678')):
            # 執行測試
            with pytest.raises(ValueError, match="Test error"):
                client.execute_partitioned_dml(
                    "UPDATE test_table SET name = 'updated' WHERE id > 100"
                )

            # 驗證記錄了錯誤
            client.logger.error.assert_called()
