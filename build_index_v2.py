import sys
from pathlib import Path
import json
import pickle
import numpy as np

# Fix imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from envision_rag.index.chunker import EnvisionChunker
from sentence_transformers import SentenceTransformer
import faiss

def build_index(scripts_dir: str, output_dir: str):
    print(f"🏗️ Building Vector Index from {scripts_dir}...")
    chunker = EnvisionChunker()
    all_chunks = []
    
    # 1. Chunking
    files = list(Path(scripts_dir).glob("*.nvn"))
    print(f"   found {len(files)} files.")
    
    for f in files:
        try:
            content = f.read_text(encoding='utf-8')
            chunks = chunker.chunk_file(content, file_path=f.name)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"⚠️ Failed to read {f.name}: {e}")

    print(f"   Generated {len(all_chunks)} chunks.")

    # Load Embedding Model
    # Changed to Multilingual model to support French Queries -> English Code
    print("🧠 Embedding chunks (using paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Load Mapping for Logical Paths
    mapping = {}
    try:
        with open("mapping.txt", "r") as f:
            for line in f:
                # Format: "12345, /path/to/logic"
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    k, v = parts[0].strip(), parts[1].strip()
                    # Mapping is Internal ID -> Logical Path
                    # Append .nvn to match filenames in chunker
                    if not k.endswith(".nvn"):
                        k = f"{k}.nvn"
                    mapping[k] = v
        print(f"🗺️ Loaded {len(mapping)} mappings.")
    except Exception as e:
        print(f"⚠️ Mapping load failed: {e}")

    # Calculate Embeddings with Logical Context
    print("Pre-processing chunks with Logical Paths...")
    texts = []
    for c in all_chunks:
        # Resolve Logical Path
        fname = Path(c.file_path).name if c.file_path else ""
        logical_path = mapping.get(fname, fname)
        
        # Inject Logical Path into Context for Embedding
        # This is CRITICAL for "Functions" matching
        enhanced_context = f"File: {logical_path}\n{c.context}"
        
        # We also treat the content as "content + context" for embedding usually
        # But here we modify what we verify? No, we modify what we ENCODE.
        # VectorTools encodes 'content' attribute? No, wait.
        # VectorTools SEARCH encodes query. Build Index encodes... text.
        # Previous code: texts = [c.content for c in all_chunks]
        # c.content ALREADY includes context (lines 97 of chunker.py: full_content = context + block)
        # So we just need to PREPEND logical path to c.content/c.context in the chunk object BEFORE valid list.
        
        # Actually, let's just make a new text list for embedding, but we need to store it?
        # No, VectorTools usually stores the Chunk object.
        # If we modify c.context/c.content here, it will be saved in pickle.
        
        c.context = f"Path: {logical_path}\n{c.context}"
        # Re-construct full content for embedding
        # The chunker constructed c.content = context + block.
        # We need to prepend the path to c.content as well for the embedding to 'see' it.
        c.content = f"Path: {logical_path}\n{c.content}"
        texts.append(c.content)

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # 3. FAISS Indexing
    print("💾 Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Cosine similarity (if normalized)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 4. Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save Index
    faiss.write_index(index, str(output_path / "faiss.index"))
    
    # Save Metadata (Chunks)
    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
        
    print(f"✅ Index saved to {output_dir}")

if __name__ == "__main__":
    build_index("env_scripts", "data/vector_store")
