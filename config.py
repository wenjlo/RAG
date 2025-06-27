import json
from enum import StrEnum

import pytz
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class ENV(StrEnum):
    LOCAL = 'local'
    RD = 'rd'
    DEV = 'dev'
    STAGING = 'staging'
    PROD = 'prod'


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    env: ENV
    platform: str = Field(min_length=1)
    etl_gcp_project: str = Field(min_length=1)

    bigquery_dw_project: str = Field(min_length=1)

    bigquery_data_mp_dataset: str = Field(min_length=1)
    bigquery_general_information_dataset: str = Field(min_length=1)
    bigquery_game_dataset: str = Field(min_length=1)

    spanner_instance: str = Field(min_length=1)
    spanner_cdp_database: str = Field(min_length=1)
    spanner_config_database: str = Field(min_length=1)

    gcs_export_bucket: str = "cdp-dev-common"

    spanner_project: str = ""
    bigquery_project: str = ""

    @property
    def default_timezone(self):
        return pytz.FixedOffset(8 * 60)

    @model_validator(mode='after')
    def modify_param(self) -> Self:
        self.spanner_project = self.etl_gcp_project
        self.bigquery_project = self.etl_gcp_project
        return self


config = Config()

if __name__ == '__main__':
    result = config.model_dump(
        mode='json',
    )
    print(json.dumps(result, indent=2))
