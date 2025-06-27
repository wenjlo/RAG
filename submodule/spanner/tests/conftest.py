from datetime import datetime
from unittest import mock

import pytest

from spanner import Client


class MockSpannerClient:
    def __init__(self, project=None):
        self.project = project

    def instance(self, instance_id):
        return MockInstance(instance_id)


class MockInstance:
    def __init__(self, instance_id):
        self.instance_id = instance_id

    def database(self, database_id, pool=None):
        return MockDatabase(database_id)


class MockDatabase:
    def __init__(self, database_id):
        self.database_id = database_id
        self.log_commit_stats = False
        self._mock_snapshot = mock.MagicMock()
        self._mock_batch = mock.MagicMock()
        self._mock_transaction = mock.MagicMock()

    def snapshot(self):
        return self._mock_snapshot

    def batch(self):
        return self._mock_batch

    def run_in_transaction(self, fn):
        return fn(self._mock_transaction)

    def execute_partitioned_dml(self, statement, params=None, param_types=None):
        # 模擬執行分割 DML 並返回受影響的行數
        return 100


class MockCommitResponse:
    def __init__(self):
        self.commit_stats = {"mutation_count": 10, "commit_timestamp": datetime.now().isoformat()}


class MockResultSet:
    def __init__(self, rows=None, field_names=None):
        self.rows = rows or []
        self._field_names = field_names or []
        self.fields = [MockField(name) for name in self._field_names]

    def __iter__(self):
        return iter(self.rows)


class MockField:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def mock_spanner_client():
    with mock.patch('google.cloud.spanner.Client', return_value=MockSpannerClient()) as mock_client:
        yield mock_client


@pytest.fixture
def client():
    """建立一個帶有模擬相依性的 Client 實例"""
    with mock.patch('google.cloud.spanner.Client', return_value=MockSpannerClient()), mock.patch(
            'google.cloud.spanner.BurstyPool', return_value=None):
        client = Client("test-project", "test-instance", "test-database")
        # 模擬 logger 以避免真實的日誌輸出
        client.logger = mock.MagicMock()
        yield client
