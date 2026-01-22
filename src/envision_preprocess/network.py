import argparse
import sys
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree

from envision_preprocess.api import EnvisionGraphAPI

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Envision Network Builder CLI")
    parser.add_argument("--build", "-b", action="store_true", help="Build the network from scripts.")
    parser.add_argument("--stats", "-s", action="store_true", help="Show network statistics.")
    parser.add_argument("--type", "-tn", type=str, help="Filter stats by Node Type (e.g. 'function').")
    parser.add_argument("--edge-type", "-te", type=str, help="Filter stats by Edge Type (e.g. 'reads').")
    parser.add_argument("--examples-count", "-n", type=int, default=3, help="Number of examples to show.")
    parser.add_argument("--query", "-q", type=str, help="Inspect a specific node ID or search for it.")
    
    # ... (rest of args) ...
    parser.add_argument("--relation", "-r", type=str, help="Filter relationships by Edge Type (e.g. 'imports', 'reads') when querying.")
    parser.add_argument("--find", "-f", action="store_true", help="Broad search for nodes matching the query (displayed as list).")
    
    # New Inspection Flags
    parser.add_argument("--globs", "-g", action="store_true", help="List all resolved glob patterns")
    parser.add_argument("--cascades", "-c", action="store_true", help="List all resolved placeholder cascades")
    
    args = parser.parse_args()
    api = EnvisionGraphAPI()
    config = api.config 
    snippet_lines = config.get("output", {}).get("snippet_lines", 10)

    # Helper: Standard Label Formatter
    def format_label(nid: str, node: dict, full_details=False) -> str:
        """
        Returns: 
        - Default: ID (Name) or ID (Path) if Name/Path != ID.
        - Full: ID (Name) [Type]  (used for headers)
        """
        name = node.get("name")
        path = node.get("path")
        
        # Priority: Path > Name > ID
        # Scripts usually have Path. Files have Path. Functions have Name.
        # User wants "ID (Name)" generically.
        # If Path is present, usage of Path as "Name" is preferred for Script/File.
        effective_name = path if path else name
        
        if effective_name and effective_name != nid:
            base = f"{nid} ({effective_name})"
        else:
            base = nid
            
        if full_details:
            return f"{base} [{node['type']}]"
        return base

    # Helper: Snippet Formatter
    def print_snippet(content: str):
        lines = content.splitlines()
        if len(lines) > (snippet_lines * 2):
            head = "\n".join(lines[:snippet_lines])
            tail = "\n".join(lines[-snippet_lines:])
            snippet = f"{head}\n\n[... {len(lines) - snippet_lines*2} lines hidden ...]\n\n{tail}"
            title = f"Content Snippet (First {snippet_lines} & Last {snippet_lines} lines)"
        else:
            snippet = "\n".join(lines)
            title = f"Content ({len(lines)} lines)"
        console.print(Panel(Syntax(snippet, "python", theme="monokai", word_wrap=True), title=title, border_style="dim"))

    # ... (Inspection Block maintained) ...
    # --- INSPECTION ---
    if args.globs or args.cascades:
        try:
            stats = api.get_stats()
        except:
            console.print("[yellow]No data. Run --build first.[/yellow]")
            return

        if args.globs:
            resolutions = stats.get("resolutions", {}).get("globs", [])
            if not resolutions:
                console.print("[yellow]No resolved glob patterns found.[/yellow]")
            else:
                table = Table(title="Resolved Glob Patterns", title_style="bold magenta", box=None)
                table.add_column("Pattern", style="cyan")
                table.add_column("Matches", style="green")
                table.add_column("Count", justify="right")
                
                for item in resolutions:
                    matches_str = "\n".join([f"- {m}" for m in item["matches"]])
                    table.add_row(item["pattern"], matches_str, str(item["count"]))
                
                console.print(Panel(table, title="✅ Resolved Globs", border_style="green"))

            # Unresolved Globs (Patterns still present in the graph)
            unresolved = [nid for nid in api._graph_cache["nodes"] if '*' in nid]
            if unresolved:
                u_table = Table(title="⚠️ Unresolved Globs (No matching files found)", title_style="bold yellow", box=None)
                u_table.add_column("Pattern Node ID", style="red")
                u_table.add_column("Reason", style="dim")
                
                for nid in unresolved:
                    u_table.add_row(nid, "No concrete file node matched this pattern in the current build.")
                    
                console.print(Panel(u_table, title="Unresolved Globs", border_style="yellow"))
            else:
                console.print("[dim]No unresolved glob patterns remaining.[/dim]")

        if args.cascades:
            resolutions = stats.get("resolutions", {}).get("placeholders", [])
            if not resolutions:
                console.print("[yellow]No resolved placeholder cascades found.[/yellow]")
            else:
                table = Table(title="Resolved Placeholder Cascades", title_style="bold magenta", box=None)
                table.add_column("Source", style="dim")
                table.add_column("Original", style="red")
                table.add_column("Resolved", style="green")
                
                for item in resolutions:
                    table.add_row(item["source"], item["original"], item["resolved"])
                    
                console.print(Panel(table, title="Cascades Detail", border_style="green"))
        return

    # --- BUILD ---
    if args.build:
        try:
            stats = api.build_graph()
            console.print(f"[green]Build Complete.[/green] Nodes: {stats.get('node_count')}, Edges: {stats.get('edge_count')}")
        except Exception as e:
             console.print(f"[red]Build Failed:[/red] {e}")
             sys.exit(1)

    # --- STATS ---
    if args.stats:
        try:
            stats = api.get_stats()
        except FileNotFoundError:
            console.print("[yellow]No data. Run --build first.[/yellow]")
            return

        # 1. Edge Type Stats
        if args.edge_type:
             target_type = args.edge_type.lower()
             console.print(f"[bold magenta]Statistics for Edge Type: {target_type}[/bold magenta]")
             
             edges = api.get_edges(target_type)
             console.print(f"Total Edges: {len(edges)}")
             
             if not edges:
                 console.print("[dim]No edges found of this type.[/dim]")
             else:
                 count = 0
                 for edge in edges:
                     if count >= args.examples_count: break
                     # Resolve Source/Target Labels for Context
                     s_node = api.get_node(edge["source"])
                     t_node = api.get_node(edge["target"])
                     
                     s_label = format_label(edge["source"], s_node) if s_node else edge["source"]
                     t_label = format_label(edge["target"], t_node) if t_node else edge["target"]
                     
                     console.print(f"Example {count+1}: {s_label} --[{target_type}]--> {t_label}")
                     if edge.get("metadata"):
                         console.print(f"   Metadata: {edge['metadata']}")
                     count += 1
                     
        # 2. Node Type Stats
        elif args.type:
            target_type = args.type.lower()
            console.print(f"[bold cyan]Statistics for Type: {target_type}[/bold cyan]")
            
            nodes_of_type = {k: v for k, v in api._graph_cache["nodes"].items() if v["type"] == target_type}
            console.print(f"Total Nodes: {len(nodes_of_type)}")
            
            if not nodes_of_type:
                console.print("[dim]No nodes found of this type.[/dim]")
            else:
                count = 0
                for nid, node in nodes_of_type.items():
                    if count >= args.examples_count: break
                    
                    # Ensure Name/Path is clearly visible in the title
                    title_label = format_label(nid, node)
                    p = Panel(json.dumps(node, indent=2), title=f"Example {count+1}: {title_label}", border_style="green")
                    console.print(p)
                    
                    if node.get("metadata", {}).get("symbols"):
                        syms = node["metadata"]["symbols"]
                        s_text = []
                        for k, v in syms.items():
                            if v: s_text.append(f"{k}: {len(v)}")
                        if s_text:
                            console.print(f"   Symbols: {', '.join(s_text)}")
                        
                    if "content" in node and node["content"]:
                        print_snippet(node["content"])
                        
                    count += 1
        
        # 3. General Stats
        else:
            # Grid
            grid = Table.grid(expand=True)
            grid.add_column()
            grid.add_column(justify="right")
            grid.add_row("[bold]Nodes[/bold]", str(stats.get("node_count", 0)))
            grid.add_row("[bold]Edges[/bold]", str(stats.get("edge_count", 0)))
            grid.add_row("[bold]Files[/bold]", str(stats.get("source_files", 0)))
            
            res = stats.get("resolutions", {})
            grid.add_row("[bold]Resolved Globs[/bold]", f"{len(res.get('globs', []))} patterns")
            grid.add_row("[bold]Resolved Cascades[/bold]", f"{len(res.get('placeholders', []))} instances")
            grid.add_row("[bold]Generated[/bold]", stats.get("generated_at", "N/A"))
            console.print(Panel(grid, title="Network Statistics", border_style="cyan"))
            
            # Nodes Table
            n_table = Table(title="Nodes by Type", box=None, expand=True)
            n_table.add_column("Type", style="bold cyan")
            n_table.add_column("Count", justify="right")
            n_table.add_column(f"Examples (Top {args.examples_count})", style="dim")
            
            nodes_by_type = stats.get("nodes_by_type", {})
            for k in sorted(nodes_by_type.keys()):
                v = nodes_by_type[k]
                examples = []
                c = 0
                for nid, n in api._graph_cache["nodes"].items():
                    if n["type"] == k:
                        label = format_label(nid, n)
                        if len(label) > 100: label = label[:97] + "..."
                        examples.append(label)
                        c += 1
                        if c >= args.examples_count: break
                n_table.add_row(k, str(v), "\n".join(examples))
            console.print(Panel(n_table, title="Nodes Structure", border_style="blue"))
            
            # Edges Table
            e_table = Table(title="Edges by Type", box=None, expand=True)
            e_table.add_column("Type", style="bold magenta")
            e_table.add_column("Count", justify="right")
            e_table.add_column("Description", style="italic")
            descriptions = {
                "reads": "Script reads a File/Table", "writes": "Script writes a File",
                "export": "Script exports a Schema/File", "imports": "Script imports another Script",
                "defines": "Script defines Function/Var/Table", "uses": "General usage",
            }
            edges_by_type = stats.get("edges_by_type", {})
            for k in sorted(edges_by_type.keys()):
                e_table.add_row(k, str(edges_by_type[k]), descriptions.get(k, ""))
            console.print(Panel(e_table, title="Edges Structure", border_style="magenta"))

    # --- QUERY ---
    if args.query:
        try:
             # FIND MODE
             if args.find:
                 matches = api.search_nodes(args.query)
                 if matches:
                     console.print(f"[green]Found {len(matches)} matches for '{args.query}':[/green]")
                     for m in matches[:20]: # Limit display
                         label = m.get("path") or m.get("name") or m["id"]
                         if label != m["id"]: label = f"{label} ({m['id']})"
                         console.print(f" - [{m['type']}] {label}")
                     if len(matches) > 20: console.print(f"... and {len(matches)-20} more.")
                 else:
                     console.print(f"[red]No matches found for '{args.query}'.[/red]")
                 return

             # EXACT/RESOLVE MODE
             node_id = api.resolve_node_id(args.query)
             
             if not node_id:
                 # Standard fuzzy suggestion fallback
                 matches = api.search_nodes(args.query)
                 if matches:
                     console.print(f"[yellow]Exact match not found. Did you mean one of these {len(matches)}?[/yellow]")
                     for m in matches[:10]:
                         label = m.get("path") or m.get("name") or m["id"]
                         console.print(f" - {label} ({m['id']})")
                     return
                 else:
                     console.print(f"[red]Node '{args.query}' not found.[/red]")
                     return

             # Node Found
             node = api.get_node(node_id)
             
             # Show JSON Structure (User Request)
             console.print(Panel(json.dumps(node, indent=2), title=f"JSON Structure: {node_id}", border_style="dim"))

             # Tree Header: ID (Name) [Type]
             header_label = format_label(node_id, node, full_details=True)
             tree = Tree(f"[bold green]{header_label}[/bold green]")
             
             # Metadata
             if "metadata" in node:
                meta = node["metadata"]
                meta_node = tree.add("Metadata")
                if "docs" in meta:
                    docs = meta["docs"]
                    doc_node = meta_node.add("Documentation")
                    for dtype, lines in docs.items():
                        if lines: doc_node.add(f"[bold]{dtype}[/bold]: {len(lines)} items")
                for k, v in meta.items():
                    if k == "symbols":
                         # Symbols Visualizer
                        sym_node = meta_node.add("Symbols")
                        for cat, items in v.items():
                            if items:
                                cat_node = sym_node.add(f"[bold]{cat}[/bold]")
                                # Sort by count desc
                                sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
                                # Show top 10
                                for key, count in sorted_items[:10]:
                                    cat_node.add(f"{key}: [cyan]{count}[/cyan]")
                                if len(sorted_items) > 10:
                                    cat_node.add(f"[dim]... and {len(sorted_items) - 10} more[/dim]")
                    elif k != "docs": 
                        meta_node.add(f"{k}: {v}")
            
             # Relationships via API
             neighbors = api.get_neighbors(node_id, relation_type=args.relation)
             
             rels_title = "Relationships"
             if args.relation: rels_title += f" (Filter: [bold magenta]{args.relation}[/bold magenta])"
             rels = tree.add(rels_title)
             
             # Outgoing
             outgoing = neighbors["outgoing"]
             if outgoing:
                 out_node = rels.add(f"Outgoing ({len(outgoing)})")
                 for item in outgoing:
                     count_str = f" [b]x{item['count']}[/b]" if item['count'] > 1 else ""
                     # Clean Format: ID (Name) - Using target_path preference for scripts/files if available
                     t_node = {"name": item.get("target_name"), "path": item.get("target_path"), "type": item.get("target_type")}
                     display = format_label(item["target_id"], t_node)
                     edge_type = f"[[bold cyan]{item['edge_type']}[/bold cyan]]"
                     
                     out_node.add(f"-----> {edge_type} {display}{count_str}")
             elif args.relation:
                 rels.add(f"[dim]No outgoing '{args.relation}' edges[/dim]")
                 
             # Incoming
             incoming = neighbors["incoming"]
             if incoming:
                 in_node = rels.add(f"Incoming ({len(incoming)})")
                 for item in incoming:
                     count_str = f" [b]x{item['count']}[/b]" if item['count'] > 1 else ""
                     # Clean Format: ID (Name)
                     s_node = {"name": item.get("source_name"), "path": item.get("source_path"), "type": item.get("source_type")}
                     display = format_label(item["source_id"], s_node)
                     edge_type = f"[[bold cyan]{item['edge_type']}[/bold cyan]]"
                     
                     in_node.add(f"<----- {edge_type} {display}{count_str}")
             elif args.relation:
                 rels.add(f"[dim]No incoming '{args.relation}' edges[/dim]")

             console.print(tree)

             if "content" in node and node["content"]:
                print_snippet(node["content"])


        except Exception as e:
            console.print(f"[red]Error querying graph:[/red] {e}")
            import traceback; traceback.print_exc()

    if not (args.build or args.stats or args.query or args.globs or args.cascades):
        parser.print_help()

if __name__ == "__main__":
    main()
