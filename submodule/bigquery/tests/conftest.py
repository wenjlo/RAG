from unittest.mock import Mock

import pandas_gbq
import pytest
from google.oauth2 import service_account
from pytest_mock import MockerFixture


@pytest.fixture
def mock_bigquery(mocker: MockerFixture) -> Mock:
    mocker_insert = mocker.patch.object(pandas_gbq, attribute='to_gbq', return_value=None)

    manager = Mock()
    manager.attach_mock(mocker_insert, 'Insert')
    return manager


@pytest.fixture
def mock_service_account_credentials(mocker: MockerFixture) -> Mock:
    mock_init = mocker.patch.object(service_account.Credentials, attribute='__init__', return_value=None)

    manager = Mock()
    manager.attach_mock(mock_init, 'Init')
    return manager
