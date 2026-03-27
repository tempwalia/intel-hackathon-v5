import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

class POCRetriever:
    def __init__(self, index_name: str = "innoscan", knowledge_base_path: str = None):
        self.index_name = index_name
        # Use sentence-transformers/all-MiniLM-L6-v2 (384-dimensional) to match Pinecone index
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        self.docsearch = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            text_key="text"  # Ensure metadata has 'text' field
        )
        # Set knowledge base path (local POC files)
        if knowledge_base_path is None:
            knowledge_base_path = Path(__file__).parent.parent / "knowledge_base" / "poc_files"
        self.knowledge_base_path = Path(knowledge_base_path)
    
    def find_similar_pocs(
        self,
        title: str,
        description: str,
        problem: str,
        score_threshold: float = 0.7,
        top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar POCs.
        Returns the top match with highest similarity score.
        """
        # Combine fields into a single query
        query_text = f"Title: {title}\nDescription: {description}\nProblem: {problem}"
        
        print(f"🔍 Querying Pinecone for similar POCs...")
        print(f"   Query: {query_text[:80]}...")
        
        # Use similarity_search_with_score to get actual similarity scores
        try:
            results = self.docsearch.similarity_search_with_score(
                query_text,
                k=top_k * 3  # Get more results to filter by POC ID
            )
        except Exception:
            # Fallback for older LangChain versions
            try:
                results = self.docsearch.similarity_search_with_relevance_scores(
                    query_text,
                    k=top_k * 3
                )
            except Exception as e:
                print(f"❌ Similarity search failed: {e}")
                return []
        
        print(f"📊 Raw results from Pinecone: {len(results)} documents")

        # Extract IDs and scores from results
        matches = []
        seen_pocs = set()
        
        for doc, score in results:
            metadata = doc.metadata or {}
            print(f"   Result: score={score}, metadata={metadata}")
            
            # Try different metadata keys for POC ID
            poc_id = (
                metadata.get("raw_id") or 
                metadata.get("poc_id") or 
                metadata.get("id") or
                "unknown"
            )
            
            # Only add if we haven't seen this POC before (avoid duplicates from chunking)
            if poc_id not in seen_pocs and score >= score_threshold:
                seen_pocs.add(poc_id)
                matches.append({
                    "poc_id": poc_id,
                    "chunk_id": metadata.get("chunk_id"),
                    "score": round(float(score), 4),
                    "title": metadata.get("title", "N/A"),
                    "raw_record": metadata.get("raw_record")
                })
                print(f"   ✅ Added match: {poc_id} (score: {score})")
                
                # Return only top 1
                if len(matches) >= 1:
                    break
        
        if not matches:
            print(f"⚠️ No matching POCs found above threshold {score_threshold}")
        
        return matches
    
    def get_poc_from_knowledge_base(self, poc_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a POC from the local knowledge base by poc_id.
        Searches for files matching the poc_id.
        """
        if not self.knowledge_base_path.exists():
            print(f"⚠️ Knowledge base path not found: {self.knowledge_base_path}")
            return None
        
        # If poc_id is "unknown" or empty, return None
        if not poc_id or poc_id == "unknown":
            print(f"⚠️ Invalid POC ID: {poc_id}")
            return None
        
        print(f"🔍 Searching for POC '{poc_id}' in knowledge base...")
        
        # Try to find the POC file
        # First try direct filename match (e.g., poc_001.json)
        for json_file in self.knowledge_base_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Check if this file contains our poc_id (check multiple ID fields)
                file_poc_id = data.get("id") or data.get("raw_id") or data.get("poc_id")
                
                if file_poc_id == poc_id:
                    print(f"✅ Found POC {poc_id} in {json_file.name}")
                    return data
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON in {json_file.name}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Error reading {json_file.name}: {e}")
                continue
        
        print(f"❌ POC '{poc_id}' not found in knowledge base at {self.knowledge_base_path}")
        
        # List available POCs for debugging
        available_pocs = []
        try:
            for json_file in self.knowledge_base_path.glob("*.json"):
                if json_file.name == "temp.json":
                    continue
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    poc_id_found = data.get("id") or data.get("raw_id")
                    available_pocs.append(poc_id_found)
                except:
                    pass
            
            if available_pocs:
                print(f"   Available POC IDs: {available_pocs}")
        except:
            pass
        
        return None
