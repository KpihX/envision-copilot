"""
Graph Builder - Parses Envision scripts and builds dependency graph.
"""
import re
import logging
from pathlib import Path
from typing import Dict

from envision_rag.graph.graph_types import DependencyGraph, Node, Edge, NodeType, EdgeType

logger = logging.getLogger(__name__)

# Constants
MAX_RECURSION_DEPTH = 10
SCRIPT_EXT = "nvn"

class GraphBuilder:
    """
    Parses .nvn files and builds the DependencyGraph.
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.graph = DependencyGraph()
        self.file_mapping = self._load_mapping()
        
        # Regex Patterns (simplified for structure extraction)
        flags = re.MULTILINE | re.IGNORECASE
        # read "/Clean/Items.ion" as Items
        self.read_pattern = re.compile(r'read\s+["\']([^"\']+)["\']', flags)
        
        # write Orders as "/Clean/Orders.ion"
        # export Orders as "/Clean/Orders.ion"
        # store Orders as "/Clean/Orders.ion"
        # write/export/store Orders as ...
        self.write_pattern = re.compile(r'(?:write|export|store)\s+\w+\s+as\s+["\']([^"\']+)["\']', flags)
        
        # show table "T" write: "file"
        self.show_write_pattern = re.compile(r'write\s*:\s*["\']([^"\']+)["\']', flags)
        
        # table Orders = ...
        self.table_pattern = re.compile(r'table\s+(\w+)\s*=', flags)

        # import "/1. utilities/Modules/Global Parameters" as GP
        self.import_pattern = re.compile(r'import\s+["\']([^"\']+)["\']', flags)
        
        # CONST resolution patterns
        self.const_decl_pattern = re.compile(r'^\s*const\s+([A-Za-z0-9_]+)\s*=\s*"(.*)"\s*$', flags)
        self.placeholder_pattern = re.compile(r'\\?\{([A-Za-z0-9_]+)\}')

    def _clean_path(self, raw_path: str) -> str:
        """Removes interpolated variables like \{storage} if they persist (fallback)."""
        # Remove \{...}
        clean = re.sub(r'\\\{[^}]+\}', '', raw_path)
        # Normalize slashes
        clean = clean.replace('\\', '/')
        # Ensure leading slash if distinct path
        if not clean.startswith('/') and '/' in clean:
             clean = '/' + clean
        return clean

    def _collect_constants(self, content: str) -> Dict[str, str]:
        consts = {}
        for line in content.splitlines():
             match = self.const_decl_pattern.match(line.strip())
             if match:
                 key, value = match.group(1), match.group(2)
                 consts[key] = self._resolve_placeholders(value, consts)
        return consts

    def _resolve_placeholders(self, text: str, consts: Dict[str, str], depth: int = 0) -> str:
        if depth > MAX_RECURSION_DEPTH:
            return text
        replaced = self.placeholder_pattern.sub(lambda match: consts.get(match.group(1), ""), text)
        if replaced == text:
            return replaced
        return self._resolve_placeholders(replaced, consts, depth=depth + 1)

    def _load_mapping(self) -> Dict[str, str]:
        """
        Loads mapping.txt to map '12345.nvn' -> '/1. utilities/...'
        """
        mapping = {}
        mapping_file = self.root_dir.parent / "mapping.txt" # Assuming root_dir is env_scripts/
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        script_id = parts[0].strip()
                        logical_path = parts[1].strip()
                        mapping[script_id] = logical_path
        return mapping

    def _get_logical_path(self, filename: str) -> str:
        """Type safe helper to get logical path"""
        return self.file_mapping.get(filename.replace(f'.{SCRIPT_EXT}', ''), filename)

    def build(self) -> DependencyGraph:
        """Scan all .nvn files and populate the graph."""
        files = list(self.root_dir.glob(f"*.{SCRIPT_EXT}"))
        print(f"🔍 GraphBuilder: Scanning {len(files)} files in {self.root_dir}...")

        for file_path in files:
            self._process_file(file_path)

        return self.graph

    def _process_file(self, file_path: Path):
        file_id = file_path.stem
        logical_path = self._get_logical_path(file_path.name)
        
        # 1. Create Script Node
        script_node = Node(
            id=logical_path, # We use the Logical Path as ID (e.g. "/1. utilities/...")
            type=NodeType.SCRIPT,
            path=str(file_path),
            metadata={"filename": file_path.name}
        )
        self.graph.add_node(script_node)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self._extract_dependencies(content, script_node)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

    def _strip_comments(self, content: str) -> str:
        """Removes // single line and /* multiline */ comments."""
        # Remove single-line comments // ...
        # (Be careful not to match // inside strings, but for now simple regex is usually enough for DSL structure)
        # Better: use a robust pattern that captures strings OR comments and selectively discards comments.
        
        # Pattern to capture strings ("..." | '...') OR comments (//... | /*...*/)
        pattern = re.compile(
            r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(/\*[^*]*\*+(?:[^/*][^*]*\*+)*/|//[^\n]*)',
            re.MULTILINE | re.DOTALL
        )

        def replacer(match):
            # If group 1 (string) matches, preserve it.
            if match.group(1):
                return match.group(1)
            # If group 2 (comment) matches, replace with space/newline to keep line numbers roughly aligned?
            # Actually, replacing with empty string is standard stripping.
            # To preserve line numbers for debug, we could replace with newlines if it's a block comment.
            return " " 

        return pattern.sub(replacer, content)

    def _extract_dependencies(self, content: str, script_node: Node):
        # 0. Strip Comments to avoid false positives (e.g. // import ...)
        content = self._strip_comments(content)

        # 1. Collect Constants
        consts = self._collect_constants(content)

        # 2. Extract READS
        for match in self.read_pattern.finditer(content):
            raw_target = match.group(1)
            resolved_target = self._resolve_placeholders(raw_target, consts)
            target_file = self._clean_path(resolved_target)
            
            # Create File Node (if not exists)
            self._ensure_file_node(target_file)
            # Create Edge: Script READS File
            self.graph.add_edge(Edge(
                source=script_node.id,
                target=target_file,
                type=EdgeType.READS,
                metadata={"context": match.group(0), "raw_path": raw_target}
            ))

        # 3. Extract WRITES
        for match in self.write_pattern.finditer(content):
            raw_target = match.group(1)
            resolved_target = self._resolve_placeholders(raw_target, consts)
            target_file = self._clean_path(resolved_target)

            self._ensure_file_node(target_file)
            # Create Edge: Script WRITES File
            self.graph.add_edge(Edge(
                source=script_node.id,
                target=target_file,
                type=EdgeType.WRITES,
                metadata={"context": match.group(0), "raw_path": raw_target}
            ))
            
        # 3b. Extract SHOW WRITES
        for match in self.show_write_pattern.finditer(content):
            raw_target = match.group(1)
            resolved_target = self._resolve_placeholders(raw_target, consts)
            target_file = self._clean_path(resolved_target)

            self._ensure_file_node(target_file)
            self.graph.add_edge(Edge(
                source=script_node.id,
                target=target_file,
                type=EdgeType.WRITES,
                metadata={"context": match.group(0), "raw_path": raw_target}
            ))
            
        # 4. Extract TABLES (Defines)
        for match in self.table_pattern.finditer(content):
            table_name = match.group(1)
            # Table Node ID: "ScriptPath::TableName" to avoid collision? 
            # Or just "TableName" if global? Envision tables are local unless read.
            # For now, let's treat them as local definitions
            node_id = f"{script_node.id}::{table_name}"
            
            table_node = Node(
                id=node_id,
                type=NodeType.TABLE,
                metadata={"name": table_name}
            )
            self.graph.add_node(table_node)
            
            self.graph.add_edge(Edge(
                source=script_node.id,
                target=node_id,
                type=EdgeType.DEFINES
            ))

        # 5. Extract IMPORTS
        for match in self.import_pattern.finditer(content):
            raw_target = match.group(1)
            resolved_target = self._resolve_placeholders(raw_target, consts)
            target_path = self._clean_path(resolved_target)
            
            # The target is likely a script too, so we treat it as a dependency
            # Ideally we check if it exists or is a "Module"
            # We add a Script Node for it if finding it? Or just verify logic later.
            # For now, treat target as SCRIPT node if it ends in .nvn or is just a path
            
            # We don't guarantee the target node exists yet, so add it as SCRIPT type optimistically?
            # Or just generic Node.
            # Usually imports are scripts.
            imported_node = Node(id=target_path, type=NodeType.SCRIPT, path=target_path, metadata={})
            self.graph.add_node(imported_node)

            self.graph.add_edge(Edge(
                source=script_node.id,
                target=target_path,
                type=EdgeType.IMPORTS,
                metadata={"context": match.group(0)}
            ))

    def _ensure_file_node(self, path: str):
        """Creates a FILE node if it doesn't exist."""
        # Clean path standardisation could happen here
        node = Node(
            id=path,
            type=NodeType.FILE,
            metadata={}
        )
        # add_node is idempotent in NetworkX usually, but our wrapper calls add_node
        # which overwrites attributes. Minimal harm here.
        self.graph.add_node(node)

if __name__ == "__main__":
    # Test run
    builder = GraphBuilder("./env_scripts")
    g = builder.build()
    print(f"✅ Graph built: {g.stats()}")
    g.save("data/dependency_graph.json")
