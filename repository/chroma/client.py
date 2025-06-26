import chromadb


class Client:
    def __init__(self):
        self.client = chromadb.PersistentClient(path='./chroma_data')
        self.collection = None
