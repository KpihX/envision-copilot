from pathlib import Path
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict, Any

class VectorTools:
    def __init__(self, index_dir: str = "data/vector_store"):
        self.index_dir = Path(index_dir)
        self.index = None
        self.chunks = []
        self.model = None
        self.reranker = None

    def _ensure_loaded(self):
        if self.index is None:
            print("⏳ Loading Vector Index...")
            index_path = self.index_dir / "faiss.index"
            meta_path = self.index_dir / "metadata.pkl"

            if not index_path.exists() or not meta_path.exists():
                raise FileNotFoundError("Vector index not found. Run build_index.py first.")

            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load Bi-Encoder for retrieval (Fast) - Logic: Multilingual for Fr <-> En match
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            
            # Load Cross-Encoder for Reranking (Accurate)
            # This model is small but significantly improves precision
            # DISABLED TEMPORARILY: ms-marco is English-Only and kills French queries.
            self.reranker = None
            # try:
            #     self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            # except Exception as e:
            #     print(f"⚠️ Could not load Reranker: {e}. Falling back to standard retrieval.")
            #     self.reranker = None

    def search_code(self, query: str, k: int = 5) -> List[str]:
        """
        Semantic search with Two-Stage Retrieval (Recall + Rerank).
        1. Recall: Fetch top 50 candidates using Bi-Encoder (FAISS).
        2. Precision: Rerank top 50 using Cross-Encoder.
        3. Return top k (default 5).
        """
        self._ensure_loaded()
        
        # 1. Recall (Broad Search)
        # Fetch more candidates (e.g. 50) to ensure we don't miss the answer
        recall_k = 50 
        
        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec).astype('float32')
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, recall_k)
        
        candidates = []
        for idx in indices[0]:
            if idx == -1: continue
            candidates.append(self.chunks[idx])
            
        if not candidates:
            # Even if no semantic candidates, we might want to return function candidates?
            # But usually we fallback to nothing. 
            # With Hybrid, we should check functions.
            candidates = []

        # 1.5 Hybrid Injection (Functions)
        # Strategy: Inject ALL function definitions to bypass Semantic Recall gaps
        func_chunks = [c for c in self.chunks if c.block_type in ['export', 'def', 'process']]
        
        # Merge Unique
        existing_contexts = set(c.content for c in candidates)
        for fc in func_chunks:
            if fc.content not in existing_contexts:
                candidates.append(fc)
                existing_contexts.add(fc.content)
        
        if not candidates:
            return []
            
        # 2. Rerank (Precision)
        if self.reranker:
            # Prepare pairs [Query, Chunk Content]
            pairs = [[query, c.context + "\n" + c.content] for c in candidates]
            scores = self.reranker.predict(pairs)
            ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            final_results = ranked_candidates[:k]
        else:
            # Fallback: Heuristic Rerank (Token Overlap)
            # This is critical for Hybrid Search when Cross-Encoder is unavailable/biased
            def score_chunk(q_tokens, chunk):
                # Normalize chunk text
                text = (chunk.context + " " + chunk.content).lower()
                # Simple score: count query tokens present in text
                score = 0
                for t in q_tokens:
                    # Basic accent handling could go here, but let's trust simple matches first
                    if t in text: score += 1
                    
                # Boost Function Definitions significantly if query mentions "function"
                if chunk.block_type in ['export', 'def', 'process']:
                     # Check if query asks for function
                     if any(x in q_tokens for x in ['function', 'fonction', 'def', 'calcul']):
                         score += 2.0
                return score

            q_tokens = query.lower().replace("?", "").replace(".", "").split()
            ranked = sorted([(c, score_chunk(q_tokens, c)) for c in candidates], key=lambda x: x[1], reverse=True)
            final_results = ranked[:k]
            
        results = []
        for chunk, score in final_results:
            snippet = f"File: {chunk.file_path}\n"
            snippet += f"Type: {chunk.block_type}\n"
            # snippet += f"Score: {score:.4f}\n" # Debug info
            snippet += f"Context:\n{chunk.context}\n"
            snippet += f"Code:\n{chunk.content}\n"
            results.append(snippet)
            
        return results

if __name__ == "__main__":
    vt = VectorTools()
    # Test Reranking
    res = vt.search_code("calculation of stock", k=3)
    for r in res:
        print(r)
        print("-" * 40)
