from typing import List, Dict, Any, Union
import json

class FactStore:
    """
    Stores verified facts extracted by tools to prevent hallucination.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.facts: List[Any] = []
        self._seen_refs = set()
        self.config = config or {}
        
    def add_fact(self, fact: Union[str, Dict, List]):
        """
        Adds a fact while attempting to deduplicate references.
        """
        if isinstance(fact, list):
            for item in fact:
                self.add_fact(item)
            return

        # Deduplication logic for Structured References
        if isinstance(fact, dict) and "source_script" in fact and "target_file" in fact:
            key = f"{fact['source_script']}|{fact.get('relationship', 'ref')}|{fact['target_file']}"
            if key in self._seen_refs:
                return
            self._seen_refs.add(key)
        
        # Deduplication for simple strings
        if isinstance(fact, str):
            key = fact.strip()
            if key in self._seen_refs:
                return
            self._seen_refs.add(key)

        self.facts.append(fact)

    def _smart_truncate(self, obj: Any, max_len: int = 100) -> Any:
        """Recursively truncates long strings in dictionaries or lists."""
        if isinstance(obj, str):
            if len(obj) > max_len:
                half = max_len // 2
                return f"{obj[:half]} ...[+{len(obj)-max_len} chars]... {obj[-half:]}"
            return obj
        elif isinstance(obj, dict):
            return {k: self._smart_truncate(v, max_len) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._smart_truncate(i, max_len) for i in obj]
        return obj

    def render_memory_appendix(self) -> str:
        """
        Returns a formatted JSON block of all facts with smart truncation.
        Used for the final transparent appendix.
        """
        if not self.facts:
            return "No facts recorded."
        
        # Get truncation limit from config
        limit = self.config.get("presentation", {}).get("max_string_len", 100)
        
        # Create a deep copy with truncation for display
        clean_facts = self._smart_truncate(self.facts, max_len=limit)
        return json.dumps(clean_facts, indent=2, ensure_ascii=False)

    def get_facts_text(self) -> str:
        """
        Returns a compressed representation for LLM Context (Not the UI).
        """
        if not self.facts:
             return ""
        
        buffer = ["\n### OBSERVED FACTS (Non-Negotiable Truths):"]
        
        for f in self.facts:
            # Format nicely for LLM comprehension
            if isinstance(f, dict):
                if "source_script" in f:
                     buffer.append(f"- [Graph] {f['source_script']} {f.get('relationship','->')} {f['target_file']}")
                elif "score" in f:
                     # RAG Result - Key info only
                     content_preview = f.get('content', '')[:100].replace('\n', ' ')
                     buffer.append(f"- [RAG] {f.get('source_id')} (Score: {f.get('score', 0):.2f}): {content_preview}...")
                else:
                     buffer.append(f"- {json.dumps(f)}")
            else:
                buffer.append(f"- {str(f)}")
                
        return "\n".join(buffer)

    def get_facts_json(self) -> List[Any]:
        return self.facts
