import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Chunk:
    content: str
    block_type: str
    start_line: int
    file_path: Optional[str] = None
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
        # Regex to detect start of blocks
        # Added: export, def, process for function definitions
        self.block_starters = re.compile(r'^\s*(read|write|export|def|process|store|import|show|table|where|keep|when)\b', re.IGNORECASE)
        # Regex to detect comments
        self.comment_line = re.compile(r'^\s*///?')

    def chunk_file(self, content: str, file_path: Optional[str] = None) -> List[Chunk]:
        lines = content.splitlines()
        chunks = []
        current_block = []
        current_start = 1
        current_type = "code"
        pending_comments = []

        for i, line in enumerate(lines):
            line_no = i + 1
            stripped = line.strip()
            
            # 1. Handle Comments (Context or Separation)
            if self.comment_line.match(stripped):
                # If we have a block accumulating and this is a Section Header (///), break it
                if current_block and stripped.startswith("///"): 
                     self._commit_chunk(chunks, current_block, current_start, current_type, pending_comments, file_path)
                     current_block = []
                     current_start = line_no
                     # pending_comments are consumed/cleared in commit_chunk usually, 
                     # but here we start a new section, so previous block is done.
                     # The pending comments *preceding* this were for the prev block.
                     # THIS comment is for the next block.
                     pending_comments = [] 

                pending_comments.append(line)
                continue

            # 2. Skip Empty Lines (but keep blocks together if inside?)
            if not stripped:
                if current_block:
                    # An empty line often signals end of block in DSL
                    self._commit_chunk(chunks, current_block, current_start, current_type, pending_comments, file_path)
                    current_block = []
                    current_start = line_no + 1
                    current_type = "code"
                    pending_comments = [] 
                continue

            # 3. Detect Block Start
            match = self.block_starters.match(stripped)
            if match:
                # If we have a running block, commit it (unless it's just 'code' and we are starting a real block)
                if current_block:
                     self._commit_chunk(chunks, current_block, current_start, current_type, pending_comments, file_path)
                     current_block = []
                     pending_comments = []
                
                current_start = line_no
                current_type = match.group(1).lower()
                current_block.append(line)
            
            # 4. Continuation
            else:
                if not current_block:
                    current_start = line_no
                    current_type = "logic"
                current_block.append(line)

        # Commit last
        if current_block:
            self._commit_chunk(chunks, current_block, current_start, current_type, pending_comments, file_path)

        return chunks

    def _commit_chunk(self, chunks, block, start_line, block_type, comments, file_path):
        # Attach comments as context found IMMEDIATELY before?
        # Actually comments list was accumulating.
        context = "\n".join(comments)
        full_content = context + "\n" + "\n".join(block) if context else "\n".join(block)
        
        # Semantic Injection: Boost function definitions
        if block_type in ['export', 'def', 'process']:
             context = f"Function Definition in {file_path or 'script'}:\n{context}"

        chunks.append(Chunk(
            content=full_content.strip(),
            block_type=block_type,
            start_line=start_line,
            file_path=file_path,
            context=context.strip()
        ))
