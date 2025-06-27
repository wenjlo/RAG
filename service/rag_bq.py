import os
from uuid import uuid4
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_google_community import BigQueryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from config import config
from repository.bigquery.bq import BQ
from repository.cloud_storage import GCS

from asset.Prompt import Prompt
from asset.ReadFiles import chunk_str

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'YOUR CREDENTIALS HERE'


class VectorDB:
    def __init__(self):
        super().__init__()
        self.llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.prompt = Prompt()
        self.bq = BQ()
        self.gcs = GCS()
        self._vectorstore = None

    @property
    def vector_store(self):
        if self._vectorstore is None:
            self._vectorstore = BigQueryVectorStore(
                project_id=self.bq._CDP_PROJECT,
                dataset_name=self.bq._CDP_RAG_DATASET,
                table_name=self.bq._rag_table,
                location="US",
                embedding=self.embedding_model
            )
        return self._vectorstore

    def insert(self, filename: str, task_id: str, text: str):
        text_id = str(uuid4())
        text_df = chunk_str(text)
        metadata = {"task_id": task_id, "text_id": text_id, "filename": filename,
                    "created_time": datetime.now(tz=config.default_timezone)}
        metadata_list = text_df[["chunk_id", "total_chunks"]].apply(
            lambda x: {"chunk_id": x["chunk_id"], "total_chunks": x["total_chunks"], **metadata}, axis=1)

        self.gcs.export_text(task_id, text_id, text)
        self.vector_store.add_texts(
            texts=text_df["document"].to_list(),
            metadatas=metadata_list
        )
        return text_id

    def delete(self, task_id: str, text_id: str):
        self._check_text_exists(task_id, text_id)
        self.gcs.delete_text(task_id, text_id)
        self.bq.delete_text_by_id(text_id)
        return "Success"

    def update(self, task_id: str, text_id: str, filename: str, text: str):
        self._check_text_exists(task_id, text_id)
        text_df = chunk_str(text)
        metadata = {"task_id": task_id, "text_id": text_id, "filename": filename,
                    "created_time": datetime.now(tz=config.default_timezone)}
        metadata_list = text_df[["chunk_id", "total_chunks"]].apply(
            lambda x: {"chunk_id": x["chunk_id"], "total_chunks": x["total_chunks"], **metadata}, axis=1)

        self.gcs.export_text(task_id, text_id, text)
        self.bq.delete_text_by_id(text_id)
        self.vector_store.add_texts(
            texts=text_df["document"].to_list(),
            metadatas=metadata_list
        )
        return "Success"

    def query(self, question: str, task_id: str):
        question_embed = self.embedding_model.embed_query(question)
        df = self.bq.get_nearest_text(question_embed, task_id)
        return ','.join([row["content"] for index, row in df.iterrows()])

    def predict(self, task_id: str, question: str):
        retriever_runnable = RunnableLambda(lambda inputs: self.query(inputs["question"], inputs["task_id"]))

        # retriever = self.vector_store.as_retriever(search_kwargs={
        #     "k": 5
        # })
        # retriever_runnable = RunnableLambda(
        #     lambda x: retriever.invoke(
        #         x["question"],
        #         filter={"task_id": x["task_id"]}
        #     )
        # )

        chain = (
                {"context": retriever_runnable, "question": lambda x: x["question"]}
                | self.prompt.prompt_template(task_id) | self.llm_model | StrOutputParser()
        )
        response = chain.invoke({
            "question": question,
            "task_id": task_id
        })
        return response

    def _check_text_exists(self, task_id: str, text_id: str):
        df = self.bq.get_text_by_id(task_id, text_id)
        if df.empty:
            raise ValueError("The document is not exists")
