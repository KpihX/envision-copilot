from typing import List, Dict, Any
import logging
from envision_rag.graph.graph_types import DependencyGraph, EdgeType, NodeType

logger = logging.getLogger(__name__)

class GraphTools:
    """
    Exposes deterministic graph queries as Tools for the Agent.
    """
    def __init__(self, graph: DependencyGraph):
        self.graph = graph

    def scan_references(self, query: str) -> List[Dict[str, Any]]:
        """
        Generic graph query to find references.
        Args:
            query: string "keyword path_pattern" (e.g. "import *" or "read Items.ion")
        """
        parts = query.strip().split(maxsplit=1)
        keyword = parts[0].lower() if parts else "any"
        path_pattern = parts[1] if len(parts) > 1 else "*"
        
        results = []
        
        # Mapping keywords to Edge Types
        edge_types = []
        if keyword in ["read", "reads"]:
            edge_types.append(EdgeType.READS.value)
        elif keyword in ["write", "writes", "export", "exports"]:
            edge_types.append(EdgeType.WRITES.value)
        elif keyword in ["import", "imports"]:
            edge_types.append(EdgeType.IMPORTS.value)
        else:
            # "any" or unknown -> check all relevant types
            edge_types = [EdgeType.READS.value, EdgeType.WRITES.value, EdgeType.IMPORTS.value]

        # Iterate edges
        for u, v, data in self.graph._graph.edges(data=True):
            if data.get('type') not in edge_types:
                continue
            
            # Check path pattern on target (v)
            # v is the file/resource being read/written/imported
            if path_pattern != "*" and path_pattern.lower() not in v.lower():
                continue
                
            results.append({
                "source_script": u,
                "relationship": data.get('type'),
                "target_file": v,
                "context": data.get('metadata', {}).get('context', '')
            })
            
        results = sorted(results, key=lambda x: x['target_file'])
        
        # Calculate unique targets for deterministic counting
        unique_targets = sorted(list(set(r['target_file'] for r in results)))
        
        return {
            "summary": f"Found {len(results)} references matching '{query}'",
            "count": len(results),
            "unique_targets_count": len(unique_targets),
            "unique_targets": unique_targets,
            "results": results,
            "query": query
        }

    def list_files(self) -> List[str]:
        """Returns all file paths known in the graph."""
        return [n for n, d in self.graph._graph.nodes(data=True) if d['type'] == NodeType.FILE.value]

    def _find_nodes(self, pattern: str, type_filter: NodeType) -> List[str]:
        """Helper to find nodes matching a pattern."""
        matches = []
        pattern = pattern.lower().strip()
        
        for node_id, data in self.graph._graph.nodes(data=True):
            if 'type' not in data:
                continue
            if data['type'] != type_filter.value:
                continue
                
            # Exact match check
            if node_id.lower() == pattern:
                matches.append(node_id)
                continue
                
            # Fuzzy/Partial match
            if pattern in node_id.lower():
                matches.append(node_id)
        
        return matches

    def describe_impact(self, script_path: str) -> Dict[str, Any]:
        """
        Trace downstream impact: Script -> Writes -> File -> Reads -> Script
        """
        script_nodes = self._find_nodes(script_path, NodeType.SCRIPT)
        if not script_nodes:
            return {"error": f"Script '{script_path}' not found."}
            
        impact = {}
        for script_id in script_nodes:
            # Find files written by this script
            written_files = self.graph._graph.successors(script_id)
            files_impacted = []
            downstream_scripts = set()
            
            for file_id in written_files:
                edge = self.graph._graph.get_edge_data(script_id, file_id)
                if edge['type'] == EdgeType.WRITES.value:
                    readers = self.graph.get_readers(file_id)
                    files_impacted.append({
                        "file": file_id,
                        "read_by": readers
                    })
                    downstream_scripts.update(readers)
            
            impact[script_id] = {
                "generated_files": len(files_impacted),
                "details": files_impacted,
                "total_downstream_scripts": len(downstream_scripts)
            }
            
        return impact
