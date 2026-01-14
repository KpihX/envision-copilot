import argparse
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree

from envision_preprocess.builder import NetworkBuilder
from envision_preprocess.utils import ConfigLoader

console = Console()

def load_data(config):
    meta_path = Path(config.get("output", {}).get("metadata_file", "data/network/metadata.json"))
    net_path = Path(config.get("output", {}).get("network_file", "data/network/network.json"))
    
    stats = {}
    graph = {"nodes": {}, "edges": []}
    
    if meta_path.exists():
        with open(meta_path, 'r') as f: stats = json.load(f)
    if net_path.exists():
        with open(net_path, 'r') as f: graph = json.load(f)
        
    return stats, graph

def main():
    parser = argparse.ArgumentParser(description="Envision Network Builder CLI")
    parser.add_argument("--build", "-b", action="store_true", help="Build the network from scripts.")
    parser.add_argument("--stats", "-s", action="store_true", help="Show network statistics.")
    parser.add_argument("--type", "-t", type=str, help="Filter stats by Node Type (e.g. 'function').")
    parser.add_argument("--examples-count", "-n", type=int, default=3, help="Number of full examples to show for type stats.")
    parser.add_argument("--query", "-q", type=str, help="Inspect a specific node ID or search for it.")
    parser.add_argument("--relation", "-r", type=str, help="Filter relationships by Edge Type (e.g. 'imports', 'reads') when querying.")
    
    args = parser.parse_args()
    config = ConfigLoader.load_config()
    
    # ... (existing build/stats logic) ...

    if args.query:
        stats, graph = load_data(config)
        query = args.query
        
        # 1. Try Exact ID Match (Logic unchanged)
        node_id = query
        node = graph["nodes"].get(node_id)
        
        # 2. Try Exact Logical Path Match (Logic unchanged)
        if not node:
            for nid, n in graph["nodes"].items():
                if n.get("metadata", {}).get("logical_path") == query:
                    node_id = nid
                    node = n
                    break
        
        # 3. Fuzzy Search / Not Found (Logic unchanged)
        if not node:
            matches = []
            for nid, n in graph["nodes"].items():
                if query.lower() in nid.lower(): matches.append(nid); continue
                lpath = n.get("metadata", {}).get("logical_path")
                if lpath and query.lower() in lpath.lower(): matches.append(lpath)
            
            if matches:
                 console.print(f"[yellow]Exact match not found. Did you mean one of these {len(matches)}?[/yellow]")
                 for m in sorted(matches)[:10]: console.print(f" - {m}")
                 return
            else:
                 console.print(f"[red]Node '{query}' not found.[/red]")
                 return
        
        # Node Found - Show Full Details
        label = node_id
        if "metadata" in node and "logical_path" in node["metadata"]:
            label = f"{node['metadata']['logical_path']} ({node_id})"
            
        tree = Tree(f"[bold green]{label}[/bold green] [[i]{node['type']}[/i]]")
        
        # Metadata (Unchanged)
        if "metadata" in node:
            meta = node["metadata"]
            meta_node = tree.add("Metadata")
            if "docs" in meta:
                docs = meta["docs"]
                doc_node = meta_node.add("Documentation")
                for dtype, lines in docs.items():
                    if lines: doc_node.add(f"[bold]{dtype}[/bold]: {len(lines)} items")
            for k, v in meta.items():
                if k != "docs": meta_node.add(f"{k}: {v}")

        # Relationships
        rels_title = "Relationships"
        if args.relation:
            rels_title += f" (Filter: [bold magenta]{args.relation}[/bold magenta])"
        rels = tree.add(rels_title)
        
        def get_node_label(nid):
            n = graph["nodes"].get(nid)
            if not n: return nid
            if n["type"] == "script": return f"{n.get('metadata', {}).get('logical_path', nid)} [dim]({nid})[/dim]"
            if n["type"] in ["table", "function", "var"]: return f"{n.get('metadata', {}).get('name', nid)} [dim]({nid})[/dim]"
            return nid

        # Outgoing
        outgoing = [e for e in graph["edges"] if e["source"] == node_id]
        if args.relation: outgoing = [e for e in outgoing if e["type"] == args.relation]
        
        if outgoing:
            out_node = rels.add(f"Outgoing ({len(outgoing)})")
            for e in outgoing:
                target_label = get_node_label(e['target'])
                out_node.add(f"---[{e['type']}]--> {target_label}")
        elif args.relation:
             rels.add(f"[dim]No outgoing '{args.relation}' edges[/dim]")
                
        # Incoming
        incoming = [e for e in graph["edges"] if e["target"] == node_id]
        if args.relation: incoming = [e for e in incoming if e["type"] == args.relation]
        
        if incoming:
            in_node = rels.add(f"Incoming ({len(incoming)})")
            for e in incoming:
                source_label = get_node_label(e['source'])
                in_node.add(f"<--[{e['type']}]--- {source_label}")
        elif args.relation:
             rels.add(f"[dim]No incoming '{args.relation}' edges[/dim]")

        console.print(tree)
        
        # Content Display (Unchanged)
        if "content" in node and node["content"]:
            console.print(Panel(Syntax(node["content"], "python", theme="monokai", word_wrap=True), title="Full Content", border_style="dim"))

    if not (args.build or args.stats or args.query):
        parser.print_help()

if __name__ == "__main__":
    main()
