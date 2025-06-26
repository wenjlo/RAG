from repository.chroma.chromadb import Chroma

chroma = Chroma()
#
for name in ['ner','sql']:
    #chroma.delete_collection(collection_name=name)
    chroma.create_collection(name)
    chroma.add_files(f'./docs/{name}',collection_name=name)