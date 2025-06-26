from typing import List
import os


def chunk_text(text: str, chunk_size: int = 256, chunk_overlap: int = 128) -> List[str]:
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


def from_txt(input_directory: str, chunk_size: int = 1024, chunk_overlap: int = 50):
    documents_to_add, metadata_to_add, ids_to_add = [], [], []
    added_count = 0
    print(f"\nProcessing files from '{input_directory}':")
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_directory, filename)
            print(f"  - Reading '{filename}'...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                # Chunk the content
                chunks = chunk_text(file_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                for i, chunk in enumerate(chunks):
                    unique_id = f"{os.path.splitext(filename)[0]}_chunk_{i + 1}"
                    documents_to_add.append(chunk)
                    metadata_to_add.append({
                        "source_file": filename,
                        "original_path": file_path,
                        "chunk_id": i + 1,
                        "total_chunks": len(chunks)
                    })
                    ids_to_add.append(unique_id)
                added_count += 1
                print(f"Added {len(chunks)} chunks from '{filename}'.")

            except FileNotFoundError:
                print(f"Error: File '{file_path}' not found.")
            except Exception as e:
                print(f"Error processing '{file_path}': {e}")
    return documents_to_add, metadata_to_add, ids_to_add
