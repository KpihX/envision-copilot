import re
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from .typedefs import Network, Node, Edge, NodeType, EdgeType
from .utils import ConfigLoader

logger = logging.getLogger(__name__)

class NetworkBuilder:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self.parsing_config = self.config.get("parsing", {})
        self.output_config = self.config.get("output", {})
        
        self.root_dir = Path(self.parsing_config.get("script_dir", "scripts"))
        self.script_ext = self.parsing_config.get("script_ext", "nvn")
        self.max_recursion = self.parsing_config.get("recursion_limit", 10)
        
        self.network = Network()
        self.file_mapping = ConfigLoader.load_mapping(self.config)
        # Create Reverse Mapping (Logical Path -> File ID)
        self.reverse_mapping = {v: k for k, v in self.file_mapping.items()}
        
        # Regex Patterns
        flags = re.MULTILINE | re.IGNORECASE
        self.read_pattern = re.compile(r'read\s+["\']([^"\']+)["\']', flags)
        self.write_pattern = re.compile(r'write\s+\w+\s+as\s+["\']([^"\']+)["\']', flags)
        # Handles:
        # 1. export MyTable as "/path"
        # 2. export schema "/path" ...
        # 3. export "/path" ...
        self.export_pattern = re.compile(r'export\s+(?:(?:\w+\s+as|schema)\s+)?["\']([^"\']+)["\']', flags)
        self.show_write_pattern = re.compile(r'write\s*:\s*["\']([^"\']+)["\']', flags)
        self.show_export_pattern = re.compile(r'export\s*:\s*["\']([^"\']+)["\']', flags)
        self.table_pattern = re.compile(r'table\s+(\w+)\s*=', flags)
        self.import_pattern = re.compile(r'import\s+["\']([^"\']+)["\']', flags)
        self.const_decl_pattern = re.compile(r'^\s*const\s+([A-Za-z0-9_]+)\s*=\s*"(.*)"\s*$', flags)
        self.placeholder_pattern = re.compile(r'\\?\{([A-Za-z0-9_]+)\}')
        
        # Function Pattern
        self.func_pattern = re.compile(r'(?:process|def)\s+([A-Za-z0-9_]+)', re.IGNORECASE)

    def _collect_constants(self, content: str) -> Dict[str, str]:
        consts = {}
        for line in content.splitlines():
             match = self.const_decl_pattern.match(line.strip())
             if match:
                 key, value = match.group(1), match.group(2)
                 consts[key] = self._resolve_placeholders(value, consts)
        return consts

    def _resolve_placeholders(self, text: str, consts: Dict[str, str], depth: int = 0) -> str:
        if depth > self.max_recursion:
            return text
        replaced = self.placeholder_pattern.sub(lambda match: consts.get(match.group(1), ""), text)
        if replaced == text:
            return replaced
        return self._resolve_placeholders(replaced, consts, depth=depth + 1)

    def _strip_comments(self, content: str) -> str:
        """Strips logic-hiding comments but keeps strings safe."""
        pattern = re.compile(
            r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|(/\*[^*]*\*+(?:[^/*][^*]*\*+)*/|//[^\n]*)',
            re.MULTILINE | re.DOTALL
        )
        return pattern.sub(lambda m: m.group(1) if m.group(1) else " ", content)

    def _extract_docs(self, content: str) -> Dict[str, List[str]]:
        """Extracts structured documentation comments."""
        docs = {
            "structure": [], # ///
            "business": [],  # //'
            "user": [],      # """
            "memos": []      # ////
        }
        
        lines = content.splitlines()
        in_markdown = False
        markdown_buf = []
        
        for line in lines:
            stripped = line.strip()
            
            # Markdown block toggle
            if stripped.startswith('"""'):
                if in_markdown:
                    # End block
                    markdown_buf.append(stripped.replace('"""', ''))
                    docs["user"].append("\n".join(markdown_buf))
                    markdown_buf = []
                    in_markdown = False
                else:
                    # Start block
                    in_markdown = True
                    markdown_buf.append(stripped.replace('"""', ''))
                continue
            
            if in_markdown:
                markdown_buf.append(line) # Keep indentation
                continue

            # Line comments
            if stripped.startswith("////"):
                docs["memos"].append(stripped[4:].strip())
            elif stripped.startswith("///"):
                docs["structure"].append(stripped[3:].strip())
            elif stripped.startswith("//'"):
                docs["business"].append(stripped[3:].strip())
                
        return docs

    def _extract_function_body(self, lines: List[str], start_idx: int) -> str:
        """Simple heuristic: Capture indented block following definition."""
        body = [lines[start_idx]] # Include definition
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip(): # Empty lines are part of body
                body.append(line)
                continue
                
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent > base_indent:
                body.append(line)
            else:
                break # End of block
        return "\n".join(body)

    def build(self):
        files = list(self.root_dir.glob(f"*.{self.script_ext}"))
        print(f"🔍 NetworkBuilder: Scanning {len(files)} files in {self.root_dir}...")
        
        for file_path in files:
            self._process_file(file_path)

        self._save_network()
        self._save_metadata(len(files))

    def _process_file(self, file_path: Path):
        logical_path = ConfigLoader.get_logical_path(file_path.name, self.file_mapping, self.script_ext)
        real_id = file_path.name.replace(f".{self.script_ext}", "")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return

        # Extract Documentation
        docs = self._extract_docs(content)
        
        # Create Script Node with FULL CONTENT
        # ID is the numeric ID. Path is the Logical Path.
        script_node = Node(
            id=real_id, 
            type=NodeType.SCRIPT,
            path=logical_path, # Logical Path here
            content=content, 
            metadata={
                "filename": file_path.name,
                "logical_path": logical_path,
                "docs": docs
            }
        )
        self.network.add_node(script_node)
        
        self._extract_dependencies(content, script_node)

    def _extract_dependencies(self, content: str, script_node: Node):
        clean_content = self._strip_comments(content)
        consts = self._collect_constants(clean_content)

        # Helper to add unique edge with metadata aggregation
        def add_unique_edge(target_id: str, edge_type: EdgeType, target_node_type: NodeType, raw_match: str):
            # Check if edge already exists for this type/target? 
            # Ideally we aggregate first then add.
            pass

        # We will aggregate targets: target_id -> {type, node_type, occurrences: []}
        # But since we have different edge types (READS vs WRITES), we aggregate by (target_id, edge_type).
        
        # Key: (target_id, edge_type) -> Value: List[str] (raw matches)
        edges_found = {}
        target_types = {} 

        # 1. READS
        for match in self.read_pattern.finditer(clean_content):
            raw = match.group(1)
            target_path = ConfigLoader.clean_path(self._resolve_placeholders(raw, consts))
            key = (target_path, EdgeType.READS)
            if key not in edges_found: edges_found[key] = []
            edges_found[key].append(raw)
            target_types[target_path] = NodeType.FILE

        # 2. WRITES
        targets = []
        for match in self.write_pattern.finditer(clean_content): targets.append((match, EdgeType.WRITES))
        for match in self.show_write_pattern.finditer(clean_content): targets.append((match, EdgeType.WRITES))
             
        for match, edge_type in targets:
            raw = match.group(1)
            target_path = ConfigLoader.clean_path(self._resolve_placeholders(raw, consts))
            key = (target_path, edge_type)
            if key not in edges_found: edges_found[key] = []
            edges_found[key].append(raw)
            target_types[target_path] = NodeType.FILE

        # 3. EXPORTS
        exports = []
        for match in self.export_pattern.finditer(clean_content): exports.append(match)
        for match in self.show_export_pattern.finditer(clean_content): exports.append(match)

        for match in exports:
            raw = match.group(1)
            target_path = ConfigLoader.clean_path(self._resolve_placeholders(raw, consts))
            key = (target_path, EdgeType.EXPORT) # Using EXPORT singular as per update
            if key not in edges_found: edges_found[key] = []
            edges_found[key].append(raw)
            target_types[target_path] = NodeType.FILE

        # 4. IMPORTS
        for match in self.import_pattern.finditer(clean_content):
            raw = match.group(1)
            raw_path = self._resolve_placeholders(raw, consts)
            clean_path = ConfigLoader.clean_path(raw_path) 
            
            # Resolve to ID
            target_id = self.reverse_mapping.get(clean_path)
            if not target_id: target_id = self.reverse_mapping.get(clean_path.lstrip('/'))
            
            if target_id:
                key = (target_id, EdgeType.IMPORTS)
                if key not in edges_found: edges_found[key] = []
                edges_found[key].append(raw)
                target_types[target_id] = NodeType.SCRIPT
            else:
                 # If not found, we skip or use path? 
                 # Let's use clean_path as ID fallback for graph completeness, but typed as SCRIPT
                 key = (clean_path, EdgeType.IMPORTS)
                 if key not in edges_found: edges_found[key] = []
                 edges_found[key].append(raw)
                 target_types[clean_path] = NodeType.SCRIPT

        # --- Create Edges and Target Nodes ---
        for (target_id, edge_type), occurrences in edges_found.items():
            # Ensure target node exists
            if target_id not in self.network.nodes:
                # We blindly create it. 
                # Note: For SCRIPT nodes, if they exist in the file list, they will be overwritten/updated 
                # by their own _process_file call later (or before).
                # `add_node` in Network usually overwrites or updates? 
                # Our Network.add_node puts into dict. So safe.
                # However, we don't want to overwrite a full SCRIPT node with a shell if we process out of order.
                # Only add if NOT exists.
                if target_id not in self.network.nodes:
                    t_node = Node(id=target_id, type=target_types[target_id])
                    self.network.add_node(t_node)
            
            # Add Unique Edge
            meta = {
                "count": len(occurrences),
                "occurrences": occurrences
            }
            # Retrieve 'raw' from first occurrence for backward compat/display if needed
            meta["raw"] = occurrences[0]
            
            self.network.add_edge(Edge(source=script_node.id, target=target_id, type=edge_type, metadata=meta))

        # 5. TABLES (DEFINES)
        # These are internal, usually unique per script.
        for match in self.table_pattern.finditer(clean_content):
            name = match.group(1)
            node_id = f"{script_node.id}::table::{name}"
            # Tables are definitions, we can overwrite.
            table_node = Node(id=node_id, type=NodeType.TABLE, metadata={"name": name})
            self.network.add_node(table_node)
            self.network.add_edge(Edge(source=script_node.id, target=node_id, type=EdgeType.DEFINES))
            
        # 6. CONSTANTS (DEFINES)
        for name, resolved_val in consts.items():
            node_id = f"{script_node.id}::const::{name}"
            const_node = Node(id=node_id, type=NodeType.VAR, metadata={"value": resolved_val, "name": name})
            self.network.add_node(const_node)
            self.network.add_edge(Edge(source=script_node.id, target=node_id, type=EdgeType.DEFINES))
            
        # 7. FUNCTIONS (DEFINES)
        lines = content.splitlines() 
        for idx, line in enumerate(lines):
            match = self.func_pattern.search(line)
            if match and not line.strip().startswith("//"):
                func_name = match.group(1)
                node_id = f"{script_node.id}::func::{func_name}"
                body = self._extract_function_body(lines, idx)
                func_node = Node(id=node_id, type=NodeType.FUNCTION, content=body, metadata={"name": func_name})
                self.network.add_node(func_node)
                self.network.add_edge(Edge(source=script_node.id, target=node_id, type=EdgeType.DEFINES))

    def _save_network(self):
        out_path = Path(self.output_config["network_file"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(self.network.to_dict(), f, indent=2)
        print(f"✅ Network saved to {out_path}")

    def _save_metadata(self, file_count: int):
        out_path = Path(self.output_config["metadata_file"])
        stats = {
            "generated_at": datetime.now().isoformat(),
            "source_files": file_count,
            "node_count": len(self.network.nodes),
            "edge_count": len(self.network.edges),
            "nodes_by_type": {},
            "edges_by_type": {}
        }
        
        for n in self.network.nodes.values():
            t = n.type.value
            stats["nodes_by_type"][t] = stats["nodes_by_type"].get(t, 0) + 1
            
        for e in self.network.edges:
            t = e.type.value
            stats["edges_by_type"][t] = stats["edges_by_type"].get(t, 0) + 1
            
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Metadata saved to {out_path}")
