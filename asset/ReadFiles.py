import pandas as pd
from typing import List
import os


def _chunk_text(text: str, chunk_size: int = 256, chunk_overlap: int = 128) -> List[str]:
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


def read_file(input_directory: str, filename: str):
    file_path = os.path.join(input_directory, filename)
    print(f"  - Reading '{filename}'...")
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    return file_content


def chunk_str(text: str, chunk_size: int = 1024, chunk_overlap: int = 50):
    chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_df = pd.DataFrame()
    for i, chunk in enumerate(chunks):
        temp = pd.DataFrame({
            "document": chunk,
            "chunk_id": i + 1,
            "total_chunks": len(chunks)
        }, index=[0])
        final_df = pd.concat([final_df, temp])

    return final_df
