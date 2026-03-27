from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from os import getenv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

get_index = getenv("index_name", "innoscan")


def get_pinecone_client() -> Pinecone:
    api_key = getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY")
    return Pinecone(api_key=api_key)
 
pc = get_pinecone_client()
index = pc.Index(get_index)
 
stats = index.describe_index_stats()
print(f"Index name:      {get_index}")
print(f"Dimension:       {stats.get('dimension', 'N/A')}")
print(f"Total vectors:   {stats.get('total_vector_count', 0)}")
print(f"Namespaces:      {list(stats.get('namespaces', {}).keys())}")