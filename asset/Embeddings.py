import chromadb
import google.generativeai as genai
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './cert/cdp-rd-vertex-ai.json'


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, model_name: str = "models/embedding-001"):
        self.model_name = model_name

    def __call__(self, content: chromadb.Documents) -> chromadb.Embeddings:
        # print(input)
        # if not isinstance(input, list) or not all(isinstance(i, str) for i in input):
        #     raise TypeError("Input to GeminiEmbeddingFunction must be a list of strings.")

        response = genai.embed_content(
            model=self.model_name,
            content=content,
            task_type="retrieval_document"
        )
        return [e for e in response['embedding']]
