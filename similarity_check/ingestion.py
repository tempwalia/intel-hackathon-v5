"""
Ingestion module for loading and processing JSON data.
Handles loading POC records, selecting columns, and chunking documents.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load_poc_json(path: str = "data/test_data") -> List[Dict[str, Any]]:
    """
    Load all POC records from all .json files in the specified folder.
    
    Args:
        path: Directory path containing .json files
    
    Returns:
        List[Dict[str, Any]]: Combined list of all records from all JSON files
    
    Raises:
        FileNotFoundError: If directory or JSON files don't exist
    """
    path_obj = Path(path)
    
    if not path_obj.is_dir():
        raise FileNotFoundError(f"Directory '{path}' not found.")
    
    all_records: List[Dict[str, Any]] = []
    json_files = sorted(path_obj.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No .json files found in '{path}'")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            # Parse records from JSON structure
            if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
                records = raw["data"]
            elif isinstance(raw, dict):
                records = [raw]
            elif isinstance(raw, list):
                records = raw
            else:
                raise ValueError("Unsupported JSON structure.")
            
            all_records.extend(records)
            print(f"✓ {json_file.name}: {len(records)} records loaded")
        
        except Exception as e:
            print(f"✗ {json_file.name}: {e}")
    
    print(f"\n✓ Total: {len(all_records)} records from {len(json_files)} files\n")
    return all_records


def select_columns(
    records: List[Dict[str, Any]], 
    include_columns: Sequence[str]
) -> List[Dict[str, Any]]:
    """
    Keep only selected columns from records. Missing keys are skipped.
    
    Args:
        records: List of record dictionaries
        include_columns: List of column names to include
    
    Returns:
        List[Dict[str, Any]]: Records with only selected columns
    """
    include_set = set(include_columns)
    return [{k: v for k, v in r.items() if k in include_set} for r in records]


def chunk_documents(
    json_data: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 20
) -> List[Document]:
    """
    Convert JSON records to chunked documents for embedding.
    
    Args:
        json_data: List of POC records/dictionaries
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List[Document]: LangChain Document objects with chunked content
    """
    if not isinstance(json_data, list):
        raise ValueError("chunk_documents expects json_data as list[dict].")

    docs: List[Document] = []
    for i, rec in enumerate(json_data):
        if not isinstance(rec, dict):
            continue

        lines = []
        for k, v in rec.items():
            if v is None:
                continue
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            lines.append(f"{k}: {v}")

        text = "\n".join(lines).strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "poc_id": str(rec.get("id", f"row_{i}")),
                    "chunk_source": "load_poc_json",
                    "raw_record": json.dumps(rec, ensure_ascii=False),
                },
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    for idx, c in enumerate(chunks):
        c.metadata = dict(c.metadata or {})
        c.metadata["chunk_id"] = idx

    print(f"✓ Created {len(chunks)} chunks from {len(docs)} documents\n")
    return chunks


def ingest_documents(
    documents: List[Document],
    index_name: str,
    embeddings
) -> object:
    """
    Ingest documents into Pinecone vector store.
    
    Args:
        documents: List of LangChain Document objects to ingest
        index_name: Name of the Pinecone index
        embeddings: HuggingFace embeddings model instance
    
    Returns:
        PineconeVectorStore: Vector store instance with ingested documents
    """
    from langchain_pinecone import PineconeVectorStore
    
    print("Debug:",)
    docsearch = PineconeVectorStore.from_documents(
        documents=documents,
        index_name=index_name,
        embedding=embeddings,
    )
    
    print(f"✓ Successfully ingested {len(documents)} documents into index '{index_name}'")
    return docsearch


# if __name__ == "__main__":
#     # Test: Load and process JSON data
#     print("Testing ingestion module...")
    
#     # Load POC records from knowledge base
#     json_path = "/Users/aditikothiyal/Code/intel/knowledge_base/poc_files"
#     json_data = load_poc_json(json_path)
#     print(f"Loaded {len(json_data)} records")
    
#     # Select specific columns
#     include_columns = [
#         "id", "title", "description", "problem", 
#         "outcome", "language", "approach", "stack", "skills"
#     ]
#     filtered_data = select_columns(json_data, include_columns)
#     print(f"Filtered to {len(filtered_data)} records with selected columns")
    
#     # Chunk the documents
#     chunks = chunk_documents(filtered_data, chunk_size=500, chunk_overlap=20)
#     print(f"✓ Total chunks: {len(chunks)}")
#     if chunks:
#         print(f"Sample chunk content:\n{chunks[0].page_content}...")
    
#     #Example: Ingest chunks into Pinecone
#     from embedding import download_hugging_face_embeddings
#     embeddings = download_hugging_face_embeddings()
#     docsearch = ingest_documents(chunks, index_name="innoscan", embeddings=embeddings)




    
