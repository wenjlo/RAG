from .client import Client
from typing import List
from asset.Embeddings import GeminiEmbeddingFunction
from asset.AddFiles import from_txt


def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> List[str]:
    """
    Splits a text into chunks with a specified size and overlap.
    For more robust chunking, consider libraries like LangChain's RecursiveCharacterTextSplitter.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += (chunk_size - chunk_overlap)
    return chunks


class Chroma(Client):
    def __init__(self):
        super().__init__()

    def create_collection(self, collection_name: str):
        self.client.create_collection(
            name=collection_name,
            embedding_function=GeminiEmbeddingFunction(),
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_search": 100,
                    "ef_construction": 100,
                    "max_neighbors": 16,
                    "num_threads": 4
                }
            })

    def add_files(self, file_path,collection_name):
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=GeminiEmbeddingFunction()
        )
        documents_to_add, metadata_to_add, ids_to_add = from_txt(file_path)
        collection.add(documents=documents_to_add, metadatas=metadata_to_add, ids=ids_to_add)

    def delete_collection(self, collection_name):
        self.client.delete_collection(collection_name)

    def retriever(self,collection,question):
        collection = self.client.get_collection(name=collection,embedding_function=GeminiEmbeddingFunction())
        result = collection.query(query_texts=[question],include=[])
        print(result)