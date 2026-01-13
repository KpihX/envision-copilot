import re
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Chunk:
    content: str
    block_type: str
    start_line: int
    context: str = ""

class EnvisionChunker:
    """
    Semantic Chunker for Envision DSL.
    Splits code into logical blocks:
    - Imports / Constants (Header)
    - Files (Read/Write)
    - Tables (Logic)
    - Show (UI)
    """
    def __init__(self):
        self.block_starters = re.compile(r'^\s*(read|write|export|import|show|table|where|keep)\b', re.IGNORECASE)
        self.comment_line = re.compile(r'^\s*///?')

    def chunk_file(self, content: str) -> List[Chunk]:
        lines = content.splitlines()
        chunks = []
        current_block = []
        current_start = 0
        current_type = "code"
        pending_comments = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 1. Handle Comments (Potential Context)
            if self.comment_line.match(stripped):
                if current_block: # End previous block if comment starts new section
                     # Heuristic: double slash might be inline, triple slash is section header
                     if stripped.startswith("///"): 
                         self._commit_chunk(chunks, current_block, current_start, current_type, pending_comments)
                         current_block = []
                         pending_comments = []
                pending_comments.append(line)
                continue

            # 2. Skip Empty
            if not stripped:
                if current_block:
                    self._commit_chunk(chunks, current_block, current_start, current_type, pending_comments)
                    current_block = []
                    # Keep pending comments for next block?
                continue

            # 3. Detect Block Start
            match = self.block_starters.match(stripped)
            if match:
                # If we have a running block, commit it
                if current_block:
                     self._commit_chunk(chunks, current_block, current_start, current_type, pending_comments)
                     current_block = []
                     # pending_comments are consumed by commit_chunk? 
                     # Actually commit_chunk should prefer using pending comments as context for the BLOCK being committed?
                     # No, pending comments usually precede the block.
                     # If we just committed the PREVIOUS block, pending comments belong to THIS new block.
                
                current_start = i + 1
                current_type = match.group(1).lower()
                current_block.append(line)
            
            # 4. Continuation (Indented or mid-block)
            else:
                if not current_block:
                    current_start = i + 1
                    current_type = "logic"
                current_block.append(line)

        # Commit last
        if current_block:
            self._commit_chunk(chunks, current_block, current_start, current_type, pending_comments)

        return chunks

    def _commit_chunk(self, chunks, block, start_line, block_type, comments):
        # Attach comments as context
        context = "\n".join(comments)
        full_content = context + "\n" + "\n".join(block) if context else "\n".join(block)
        chunks.append(Chunk(
            content=full_content,
            block_type=block_type,
            start_line=start_line,
            context=context
        ))
        comments.clear() # Consumed

if __name__ == "__main__":
    sample = """
/// Header
import "/Library" as Lib

/// Read items
read "/Items.ion" as Items with
  Sku as Id
  Name

// Calculate stock
Items.Stock = Items.OnHand + Items.OnOrder

show table "Results" with
  Items.Id
  Items.Stock
"""
    chunker = EnvisionChunker()
    result = chunker.chunk_file(sample)
    for c in result:
        print(f"[{c.block_type}:{c.start_line}]")
        print(c.content)
        print("-" * 20)
