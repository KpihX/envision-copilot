import re
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import fnmatch

from .typedefs import Network, Node, Edge, NodeType, EdgeType
from .utils import ConfigLoader
from .extractor import SymbolExtractor

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
        
        # Resolution Tracking
        self.info_resolutions = {
            "globs": [],        # List of {pattern: str, matches: List[str], count: int}
            "placeholders": []  # List of {original: str, resolved: str, source: str}
        }
        
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
        
        # Helper Regex
        self.placeholder_pattern = re.compile(r'\\?\{([A-Za-z0-9_]+)\}')
        
        # Function Pattern: Capture everything until '(', '=', or '{'
        self.func_pattern = re.compile(r'(?:process|def)\s+([^{=(]+)', re.IGNORECASE)
        
        # Variable Collection Patterns
        # 1. const Name = "Value"
        # 2. Name = "Value" (String assignment)
        # Note: We only care about string literals for path resolution.
        self.const_decl_pattern = re.compile(r'const\s+([A-Za-z0-9_]+)\s*=\s*"(.*)"')
        self.var_decl_pattern = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"(.*)"')

    def _collect_constants(self, content: str, script_id: str = "unknown") -> Dict[str, Tuple[str, int]]:
        """
        Collects 'const' definitions AND string variable assignments to resolve placeholders.
        Returns map: Name -> (ResolvedValue, LineNumber)
        """
        consts = {}
        
        for idx, line in enumerate(content.splitlines()):
             line = line.strip()
             if not line: continue
             
             # Check for 'const' declaration
             match = self.const_decl_pattern.match(line)
             if not match:
                 # Check for simple string assignment (e.g. exportPath = "...")
                 match = self.var_decl_pattern.match(line)
                 
             if match:
                 key, value = match.group(1), match.group(2)
                 
                 # Resolve using currently known values (top-down)
                 current_values = {k: v[0] for k,v in consts.items()}
                 resolved_val = self._resolve_placeholders(value, current_values)
                 
                 # Track resolution if changed
                 if value != resolved_val:
                     self.info_resolutions["placeholders"].append({
                         "original": value,
                         "resolved": resolved_val,
                         "source": f"{script_id}::var::{key}"
                     })
                 
                 consts[key] = (resolved_val, idx + 1)
                 
        return consts

    def _resolve_placeholders(self, text: str, consts: Dict[str, str], depth: int = 0) -> str:
        if depth > self.max_recursion:
            return text
        # If not found, keep the original placeholder (match.group(0))
        replaced = self.placeholder_pattern.sub(lambda match: consts.get(match.group(1), match.group(0)), text)
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

    def _extract_function_body(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """Returns (body_content, end_line_number_1_indexed)"""
        body = [lines[start_idx]] # Include definition
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        last_idx = start_idx
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip(): # Empty lines are part of body
                body.append(line)
                last_idx = i
                continue
                
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent > base_indent:
                body.append(line)
                last_idx = i
            else:
                break # End of block
        return "\n".join(body), last_idx + 1

    def build(self):
        files = list(self.root_dir.glob(f"*.{self.script_ext}"))
        print(f"🔍 NetworkBuilder: Scanning {len(files)} files in {self.root_dir}...")
        
        for file_path in files:
            self._process_file(file_path)

        # Post-Processing: Resolve Glob Nodes
        # Post-Processing: Resolve Glob Nodes
        self._resolve_glob_nodes()

        cascade_count = len(self.info_resolutions["placeholders"])
        if cascade_count > 0:
            print(f"🔄 Resolved {cascade_count} placeholder cascades.")

        self._save_network()
        self._save_metadata(len(files))

    def _resolve_glob_nodes(self):
        """
        Identify nodes with '*' (glob patterns), find matching concrete nodes,
        redirect edges, and remove the glob node.
        """
        # Snapshot keys to allow modification
        all_node_ids = list(self.network.nodes.keys())
        glob_nodes = [nid for nid in all_node_ids if '*' in nid]
        
        # We only match against FILE nodes (usually)
        candidate_nodes = [
            nid for nid in all_node_ids 
            if '*' not in nid and self.network.nodes[nid].type == NodeType.FILE
        ]
        
        resolved_count = 0
        
        for glob_id in glob_nodes:
            matches = [cand for cand in candidate_nodes if fnmatch.fnmatch(cand, glob_id)]
            
            if not matches:
                continue
            
            # Record Stat
            self.info_resolutions["globs"].append({
                "pattern": glob_id,
                "matches": matches,
                "count": len(matches)
            })
                
            # Redirect Edges
            edges_to_remove = []
            edges_to_add = []
            
            for edge in self.network.edges:
                if edge.target == glob_id:
                    edges_to_remove.append(edge)
                    for match_id in matches:
                        new_meta = edge.metadata.copy() if edge.metadata else {}
                        new_meta["glob_source"] = glob_id
                        
                        edges_to_add.append(Edge(
                            source=edge.source,
                            target=match_id,
                            type=edge.type,
                            metadata=new_meta
                        ))
            
            if edges_to_add:
                # Apply changes
                for e in edges_to_remove:
                    self.network.remove_edge(e)
                for e in edges_to_add:
                    self.network.add_edge(e)
                
                # Remove the original glob node
                self.network.nodes.pop(glob_id, None)
                resolved_count += 1
                
        if resolved_count > 0:
            print(f"✨ Resolved {resolved_count} glob patterns to concrete files.")
 
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
        script_node = Node(
            id=real_id, 
            type=NodeType.SCRIPT,
            name=Path(logical_path).name, # Use Logical Name (e.g. MyScript) not ID
            path=logical_path, 
            content=content, 
            metadata={
                # logical_path and filename removed as redundant
                "docs": docs,
                "symbols": SymbolExtractor.extract(self._strip_comments(content))
            }
        )
        self.network.add_node(script_node)
        
        self._extract_dependencies(content, script_node)

    def _extract_dependencies(self, content: str, script_node: Node):
        
        def get_line_num(match_obj):
            return content[:match_obj.start()].count('\n') + 1

        clean_content = self._strip_comments(content)
        # Pass script ID for tracking
        consts_map = self._collect_constants(clean_content, script_node.id) 
        consts_values = {k: v[0] for k, v in consts_map.items()}

        edges_found = {} 
        target_types = {} 

        # Helper to process raw match
        def process_match(raw, edge_type, target_type=NodeType.FILE):
            # Resolve placeholders
            raw_path = self._resolve_placeholders(raw, consts_values)
            
            # Track Resolution
            if raw != raw_path:
                 self.info_resolutions["placeholders"].append({
                     "original": raw,
                     "resolved": raw_path,
                     "source": f"{script_node.id} (Edge)"
                 })
            
            # 2. Treat Interpolation {Var} as Glob *
            # If path contains {..} (leftover interpolation), replace with *
            if '{' in raw_path and '}' in raw_path:
                import re
                raw_path = re.sub(r'\{[^}]+\}', '*', raw_path)
            
            # Clean Path (Aggressive)
            target_path = ConfigLoader.clean_path(raw_path)
            
            key = (target_path, edge_type)
            if key not in edges_found: edges_found[key] = []
            edges_found[key].append(raw)
            target_types[target_path] = target_type

        # 1. READS
        for match in self.read_pattern.finditer(clean_content):
            process_match(match.group(1), EdgeType.READS, NodeType.FILE)

        # 2. WRITES
        targets = []
        for match in self.write_pattern.finditer(clean_content): targets.append((match, EdgeType.WRITES))
        for match in self.show_write_pattern.finditer(clean_content): targets.append((match, EdgeType.WRITES))
        for match, type_ in targets:
            process_match(match.group(1), type_, NodeType.FILE)

        # 3. EXPORTS
        exports = []
        for match in self.export_pattern.finditer(clean_content): exports.append(match)
        for match in self.show_export_pattern.finditer(clean_content): exports.append(match)
        for match in exports:
            process_match(match.group(1), EdgeType.EXPORT, NodeType.FILE)

        # 4. IMPORTS
        for match in self.import_pattern.finditer(clean_content):
            raw = match.group(1)
            raw_path = self._resolve_placeholders(raw, consts_values)
            
            if raw != raw_path:
                 self.info_resolutions["placeholders"].append({
                     "original": raw,
                     "resolved": raw_path,
                     "source": f"{script_node.id} (Import)"
                 })
            
            clean_path = ConfigLoader.clean_path(raw_path) 
            
            target_id = self.reverse_mapping.get(clean_path)
            if not target_id: target_id = self.reverse_mapping.get(clean_path.lstrip('/'))
            
            if target_id:
                # Key using ID
                key = (target_id, EdgeType.IMPORTS)
                if key not in edges_found: edges_found[key] = []
                edges_found[key].append(raw)
                target_types[target_id] = NodeType.SCRIPT
            else:
                 # Fallback to path
                 key = (clean_path, EdgeType.IMPORTS)
                 if key not in edges_found: edges_found[key] = []
                 edges_found[key].append(raw)
                 target_types[clean_path] = NodeType.SCRIPT

        # --- Create Edges and Target Nodes ---
        for (target_id, edge_type), occurrences in edges_found.items():
            if target_id not in self.network.nodes:
                 if target_id not in self.network.nodes:
                    # If it's a file, ensure it looks like a file path?
                    name = Path(target_id).name if '/' in target_id else target_id
                    t_node = Node(id=target_id, type=target_types[target_id], name=name)
                    self.network.add_node(t_node)
            
            meta = {
                "count": len(occurrences),
                "occurrences": occurrences,
                "raw": occurrences[0]
            }
            self.network.add_edge(Edge(source=script_node.id, target=target_id, type=edge_type, metadata=meta))

        # 5. TABLES
        # These are internal, usually unique per script.
        for match in self.table_pattern.finditer(clean_content):
            name = match.group(1)
            node_id = f"{script_node.id}::table::{name}"
            lineno = get_line_num(match)
            table_node = Node(id=node_id, type=NodeType.TABLE, name=name, start_line=lineno, end_line=lineno)
            self.network.add_node(table_node)
            self.network.add_edge(Edge(source=script_node.id, target=node_id, type=EdgeType.DEFINES))
            
        # 6. CONSTANTS
        for name, (resolved_val, lineno) in consts_map.items():
            node_id = f"{script_node.id}::const::{name}"
            # Clean Metadata: Name is arg, Content is Value.
            const_node = Node(id=node_id, type=NodeType.VAR, name=name, content=resolved_val, start_line=lineno, end_line=lineno)
            self.network.add_node(const_node)
            self.network.add_edge(Edge(source=script_node.id, target=node_id, type=EdgeType.DEFINES))
            
        # 7. FUNCTIONS
        lines = content.splitlines() 
        for idx, line in enumerate(lines):
            match = self.func_pattern.search(line)
            if match and not line.strip().startswith("//"):
                full_match = match.group(1).strip()
                # Split by whitespace
                parts = full_match.split()
                if not parts: continue
                
                # Logic: Last element is name. Preceding are qualifiers.
                # Example: "def pure PonderationScale" -> parts=["pure", "PonderationScale"] (def is skipped by group 1?)
                # Wait, Regex: `(?:process|def)\s+([^{=(]+)`
                # If "def pure Name", regex matches "def " then group 1 is "pure Name".
                # parts = "pure Name".split() -> ["pure", "Name"]
                # Name = "Name", Qualifiers = ["pure"]
                # If "process Name", group 1 is "Name". parts=["Name"]. Qualifiers=[].
                # This logic holds.
                
                func_name = parts[-1] 
                qualifiers = parts[:-1]
                
                node_id = f"{script_node.id}::func::{func_name}"
                body, end_lineno = self._extract_function_body(lines, idx)
                start_lineno = idx + 1
                
                meta = {}
                # Store qualifiers. User said "process est un qualificatif".
                if qualifiers: meta["qualifiers"] = qualifiers
                
                func_node = Node(
                    id=node_id, 
                    type=NodeType.FUNCTION, 
                    name=func_name, 
                    content=body, 
                    start_line=start_lineno,
                    end_line=end_lineno,
                    metadata=meta
                )
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
            "edges_by_type": {},
            "resolutions": self.info_resolutions # Save Resolutions
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
