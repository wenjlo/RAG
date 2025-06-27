from datetime import datetime
from typing import Optional
from google.cloud import storage

from google.api_core.exceptions import NotFound
from config import config


class GCS:
    _PROJECT = config.etl_gcp_project
    _BUCKET_NAME = config.gcs_export_bucket

    _bucket: Optional[storage.Bucket] = None

    def __init__(self):
        pass

    @property
    def bucket(self):
        if GCS._bucket is None:
            client = storage.Client(project=GCS._PROJECT)
            GCS._bucket = client.get_bucket(GCS._BUCKET_NAME)
        return GCS._bucket

    @staticmethod
    def get_current_datetime_string() -> str:
        return datetime.now(tz=config.default_timezone).strftime('%Y%m%d%H%M%S')

    @staticmethod
    def get_blob_url(blob: storage.Blob) -> str:
        return f'https://storage.cloud.google.com/{blob.bucket.name}/{blob.name}'

    def export_upload_log(self, task_id: str, text_id: str, text: str) -> str:
        datetime_string = self.get_current_datetime_string()
        blob = self.bucket.blob(f'rag/log/{task_id}/{text_id}/{datetime_string}.txt')
        blob.upload_from_string(text)
        return self.get_blob_url(blob)

    def export_upload_text(self, task_id: str, text_id: str, text: str) -> str:
        blob = self.bucket.blob(f'rag/final/{task_id}/{text_id}.txt')
        blob.upload_from_string(text, content_type="text/plain")
        return self.get_blob_url(blob)

    def export_text(self, task_id: str, text_id: str, text: str):
        self.export_upload_log(task_id, text_id, text)
        self.export_upload_text(task_id, text_id, text)

    def delete_text(self, task_id: str, text_id: str):
        self.delete_text_final(task_id, text_id)
        self.delete_text_log(task_id, text_id)

    def delete_text_log(self, task_id: str, text_id: str):
        prefix = f'rag/log/{task_id}/{text_id}'
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            print(f"Deleting {blob.name}")
            self._delete_blob(blob)

    def delete_text_final(self, task_id: str, text_id: str):
        blob = self.bucket.blob(f'rag/final/{task_id}/{text_id}.txt')
        self._delete_blob(blob)

    def _delete_blob(self, blob):
        blob.reload()
        generation_match_precondition = blob.generation
        try:
            blob.delete(if_generation_match=generation_match_precondition)
        except NotFound:
            pass
