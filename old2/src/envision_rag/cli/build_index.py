"""
Envision RAG - Index Builder CLI.
Builds the graph index (NetworkX) and/or vector index (FAISS).

Usage:
    uv run build --graph           Build graph index only
    uv run build --vector          Build vector index only
    uv run build --all             Build both (default)
    uv run build --stats           Show detailed statistics
"""
import argparse
import yaml
import time
from pathlib import Path
import pickle
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn


console = Console()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_graph(config: dict, verbose: bool = True, show_stats: bool = False):
    """Build the NetworkX dependency graph."""
    from envision_rag.graph.builder import GraphBuilder
    
    scripts_dir = config.get("system", {}).get("scripts_dir", "./scripts")
    data_dir = config.get("system", {}).get("data_dir", "./data")
    graph_path = Path(data_dir) / "dependency_graph.json"
    
    if verbose:
        console.print(Panel("🏗️ Building Graph Index", style="bold blue"))
        console.print(f"   Source: [cyan]{scripts_dir}[/cyan]")
    
    start_time = time.time()
    builder = GraphBuilder(scripts_dir)
    graph = builder.build()
    build_time = time.time() - start_time
    
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    graph.save(str(graph_path))
    
    stats = graph.stats()
    
    if verbose:
        console.print(f"   ✅ Saved: [green]{graph_path}[/green]")
        console.print(f"   ⏱️ Time: [yellow]{build_time:.2f}s[/yellow]")
        console.print(f"   📊 {stats['nodes']} nodes, {stats['edges']} edges")
    
    if show_stats:
        _show_graph_stats(graph, verbose)
    
    return graph_path, build_time


def _show_graph_stats(graph, verbose: bool = True):
    """Show detailed graph statistics."""
    from envision_rag.graph.graph_types import NodeType, EdgeType
    
    console.print(Panel.fit("[bold]📊 Graph Statistics[/bold]", border_style="blue"))
    
    # Node types
    node_table = Table(title="Nodes by Type", show_header=True)
    node_table.add_column("Type", style="cyan")
    node_table.add_column("Count", justify="right")
    
    nodes = list(graph._graph.nodes(data=True))
    scripts = [n for n, d in nodes if d.get('type') == NodeType.SCRIPT.value]
    files = [n for n, d in nodes if d.get('type') == NodeType.FILE.value]
    
    node_table.add_row("Scripts", str(len(scripts)))
    node_table.add_row("Files", str(len(files)))
    console.print(node_table)
    
    # Edge types
    edge_table = Table(title="Edges by Type", show_header=True)
    edge_table.add_column("Type", style="cyan")
    edge_table.add_column("Count", justify="right")
    
    edges = list(graph._graph.edges(data=True))
    reads = [e for e in edges if e[2].get('type') == EdgeType.READS.value]
    writes = [e for e in edges if e[2].get('type') == EdgeType.WRITES.value]
    imports = [e for e in edges if e[2].get('type') == EdgeType.IMPORTS.value]
    
    edge_table.add_row("Reads", str(len(reads)))
    edge_table.add_row("Writes", str(len(writes)))
    edge_table.add_row("Imports", str(len(imports)))
    console.print(edge_table)
    
    # Sample data
    console.print(Panel.fit("[bold]📝 Sample Data[/bold]", border_style="dim"))
    console.print("[dim]Sample Scripts:[/dim]")
    for s in scripts[:3]:
        console.print(f"  • {s}")
    if len(scripts) > 3:
        console.print(f"  [dim]... and {len(scripts) - 3} more[/dim]")
    
    console.print("\n[dim]Sample Files:[/dim]")
    for f in files[:3]:
        console.print(f"  • {f}")
    if len(files) > 3:
        console.print(f"  [dim]... and {len(files) - 3} more[/dim]")


def build_vector(config: dict, verbose: bool = True, show_stats: bool = False):
    """Build the FAISS vector index."""
    from envision_rag.index.chunker import EnvisionChunker
    from sentence_transformers import SentenceTransformer
    import faiss
    
    # Load config values
    system_conf = config.get("system", {})
    index_conf = config.get("index", {})
    
    scripts_dir = system_conf.get("scripts_dir", "./scripts")
    extension = system_conf.get("file_extension", "nvn")
    mapping_file = system_conf.get("mapping_file", "mapping.txt")
    data_dir = system_conf.get("data_dir", "./data")
    
    output_dir = Path(index_conf.get("store_path", "data/vector_store"))
    embedding_model = index_conf.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if verbose:
        console.print(Panel("🧠 Building Vector Index", style="bold purple"))
        console.print(f"   Source: [cyan]{scripts_dir}/*.{extension}[/cyan]")
        console.print(f"   Model:  [dim]{embedding_model}[/dim]")
    
    start_time = time.time()
    
    # 1. Chunk files
    chunker = EnvisionChunker()
    all_chunks = []
    
    files = list(Path(scripts_dir).glob(f"*.{extension}"))
    if verbose:
        console.print(f"   Found [yellow]{len(files)}[/yellow] files")
    
    for f in files:
        try:
            content = f.read_text(encoding='utf-8')
            chunks = chunker.chunk_file(content, file_path=f.name)
            all_chunks.extend(chunks)
        except Exception as e:
            pass
    
    if verbose:
        console.print(f"   Generated [yellow]{len(all_chunks)}[/yellow] chunks")
    
    # 2. Load mapping
    mapping = {}
    try:
        with open(mapping_file, "r") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    k = parts[0].strip() + f".{extension}"
                    mapping[k] = parts[1].strip()
    except Exception:
        pass
    
    # 3. Embed
    if verbose:
        console.print("   🔄 Embedding chunks...")
    
    model = SentenceTransformer(embedding_model)
    
    texts = []
    for c in all_chunks:
        fname = Path(c.file_path).name if c.file_path else ""
        logical_path = mapping.get(fname, fname)
        c.context = f"Path: {logical_path}\n{c.context}"
        c.content = f"Path: {logical_path}\n{c.content}"
        texts.append(c.content)
    
    embeddings = model.encode(texts, show_progress_bar=verbose)
    embeddings = np.array(embeddings).astype('float32')
    
    # 4. Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    build_time = time.time() - start_time
    
    # 5. Save
    output_dir.mkdir(exist_ok=True, parents=True)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    with open(output_dir / "metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    
    if verbose:
        console.print(f"   ✅ Saved: [green]{output_dir}[/green]")
        console.print(f"   ⏱️ Time: [yellow]{build_time:.2f}s[/yellow]")
        console.print(f"   📊 {len(all_chunks)} chunks, {dimension}d vectors")
    
    if show_stats:
        _show_vector_stats(all_chunks, dimension)
    
    return output_dir, build_time


def _show_vector_stats(chunks, dimension: int):
    """Show detailed vector index statistics."""
    console.print(Panel.fit("[bold]📊 Vector Index Statistics[/bold]", border_style="purple"))
    
    # Chunk types distribution
    type_counts = {}
    for c in chunks:
        t = c.block_type
        type_counts[t] = type_counts.get(t, 0) + 1
    
    type_table = Table(title="Chunks by Type", show_header=True)
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", justify="right")
    type_table.add_column("%", justify="right")
    
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(chunks) * 100
        type_table.add_row(t, str(count), f"{pct:.1f}%")
    console.print(type_table)
    
    # Sample chunks
    console.print(Panel.fit("[bold]📝 Sample Chunks[/bold]", border_style="dim"))
    for i, c in enumerate(chunks[:2]):
        console.print(f"[dim]Chunk {i+1} ({c.block_type}):[/dim]")
        preview = c.content[:150].replace("\n", " ")
        console.print(f"  [italic]{preview}...[/italic]")


def main():
    parser = argparse.ArgumentParser(
        prog="build",
        description="Build indexes for the Envision RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run build                Build both indexes
  uv run build -g             Build graph index only
  uv run build -v             Build vector index only
  uv run build -s             Show detailed statistics
  uv run build -g -s          Build graph with stats

Configuration (config.yaml):
  scripts_dir: source folder (default: ./scripts)
  file_extension: script extension (default: nvn)
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-g", "--graph",
        action="store_true",
        help="Build graph index only (NetworkX)"
    )
    group.add_argument(
        "-v", "--vector",
        action="store_true",
        help="Build vector index only (FAISS)"
    )
    group.add_argument(
        "-a", "--all",
        action="store_true",
        default=True,
        help="Build both indexes (default)"
    )
    parser.add_argument(
        "-s", "--stats",
        action="store_true",
        help="Show detailed statistics after build"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        console.print(Panel.fit(
            "[bold]Envision RAG Index Builder[/bold]",
            border_style="blue"
        ))
    
    config = load_config()
    
    # Determine what to build
    build_g = args.graph or (not args.graph and not args.vector)
    build_v = args.vector or (not args.graph and not args.vector)
    
    total_time = 0
    
    if build_g:
        _, t = build_graph(config, verbose, args.stats)
        total_time += t
        if verbose:
            console.print()
    
    if build_v:
        _, t = build_vector(config, verbose, args.stats)
        total_time += t
    
    if verbose:
        console.print(f"\n[bold green]✅ Build complete![/bold green] (Total: {total_time:.2f}s)")


if __name__ == "__main__":
    main()
