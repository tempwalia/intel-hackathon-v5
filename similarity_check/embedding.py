"""
Embedding module for managing HuggingFace embeddings.
Handles downloading and initializing embedding models.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # fallback for older LangChain installs
    from langchain.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()


def download_hugging_face_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Download and initialize HuggingFace embeddings model.
    
    Args:
        model_name: The HuggingFace model name to use for embeddings.
                   Default: 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions)
    
    Returns:
        HuggingFaceEmbeddings: Initialized embeddings model
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"✓ Embeddings model loaded: {model_name}")
    return embeddings


def get_embedding_dimension(embeddings) -> int:
    """
    Get the dimension of the embeddings model by creating a test embedding.
    
    Args:
        embeddings: HuggingFaceEmbeddings instance
    
    Returns:
        int: Dimension of the embedding vectors
    """
    sample_vec = embeddings.embed_query("dimension check")
    return len(sample_vec)


# if __name__ == "__main__":
#     # Test: Load embeddings and check dimension
#     print("Testing embedding module...")
    
#     embeddings = download_hugging_face_embeddings()
#     dim = get_embedding_dimension(embeddings)
#     print(f"✓ Embedding dimension: {dim}")
    
#     # Test embedding a sample text
#     test_text = "What is Intelligent Ticket Classifier?"
#     test_embedding = embeddings.embed_query(test_text)
#     print(f"✓ Sample embedding created with {len(test_embedding)} dimensions")
