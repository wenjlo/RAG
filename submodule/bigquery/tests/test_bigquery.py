from unittest.mock import Mock

import pandas as pd
import pandas_gbq
from google.oauth2 import service_account
from pytest_mock import MockerFixture

import bigquery


def test_insert(mock_bigquery: Mock, mock_service_account_credentials: Mock):
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': ['1', '2', '3', '4']
    })
    execute_project_id = 'execute_project_id'
    credentials = service_account.Credentials()
    table = 'project.dataset.table'

    client = bigquery.Client(execute_project_id, credentials)
    client.insert(table, df)

    call_array = mock_bigquery.Insert.call_args.args
    assert call_array[0].equals(df)

    call_dict = mock_bigquery.Insert.call_args.kwargs
    assert call_dict['credentials'] == credentials
    assert call_dict['destination_table'] == table


def test_insert_invalid_table(mock_bigquery: Mock, mock_service_account_credentials: Mock):
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': ['1', '2', '3', '4']
    })
    execute_project_id = 'execute_project_id'
    credentials = service_account.Credentials()
    table = 'dataset.table'

    try:
        client = bigquery.Client(execute_project_id, credentials)
        client.insert(table, df)
    except Exception as e:
        assert str(
            e) == "invalid bigquery table format, insert table: dataset.table, required format: [project.dataset.table]"


def test_get(mocker: MockerFixture, mock_bigquery: Mock, mock_service_account_credentials: Mock):
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': ['1', '2', '3', '4']
    })
    mocker_get = mocker.patch.object(pandas_gbq, attribute='read_gbq', return_value=df)

    execute_project_id = 'execute_project_id'
    credentials = service_account.Credentials()
    query = 'select 1'

    client = bigquery.Client(execute_project_id, credentials)
    client.get(query)

    call_array = mocker_get.call_args.args
    assert call_array[0] == query

    call_dict = mocker_get.call_args.kwargs
    assert call_dict['project_id'] == execute_project_id
    assert call_dict['credentials'] == credentials


def test_execute_dml_operation(mocker: MockerFixture, mock_bigquery: Mock, mock_service_account_credentials: Mock):
    mocker_execute_dml_operation = mocker.patch.object(pandas_gbq, attribute='read_gbq', return_value=None)

    execute_project_id = 'execute_project_id'
    credentials = service_account.Credentials()
    operation = 'delete 1'
    expected_operation = 'delete 1; select 1;'

    client = bigquery.Client(execute_project_id, credentials)
    client.execute_dml_operation(operation)

    call_array = mocker_execute_dml_operation.call_args.args
    assert call_array[0] == expected_operation

    call_dict = mocker_execute_dml_operation.call_args.kwargs
    assert call_dict['project_id'] == execute_project_id
    assert call_dict['credentials'] == credentials
