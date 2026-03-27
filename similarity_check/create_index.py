from pinecone import Pinecone


index_name = "innoscan"  # change if desired
target_dim = 384


def index_exists(client: Pinecone, name: str) -> bool:
    indexes = client.list_indexes()
    if hasattr(indexes, "names"):  # newer SDK response object
        return name in indexes.names()
    # older/alternate response formats
    return name in [
        i["name"] if isinstance(i, dict) else getattr(i, "name", None)
        for i in indexes
    ]
