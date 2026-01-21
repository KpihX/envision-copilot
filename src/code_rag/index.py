import argparse
import json
import logging
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .indexer import GraphIndexer
from .retriever import GraphRetriever

logging.basicConfig(level=logging.ERROR) # Suppress internal logs for clean CLI
console = Console()

def main():
    parser = argparse.ArgumentParser(description="Envision Code RAG CLI")
    
    parser.add_argument("-b", "--build", action="store_true", help="Build/Rebuild the Index")
    parser.add_argument("-q", "--query", type=str, help="Search the index")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("-r", "--recall", type=int, default=None, help="Number of candidates to retrieve from FAISS")
    parser.add_argument("--no-rerank", "-n", action="store_true", help="Disable reranking")
    parser.add_argument("-s", "--stats", action="store_true", help="Show index statistics")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of sample chunks to show with --stats")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if args.build:
        print("🏗️ Triggering Index Build...")
        indexer = GraphIndexer()
        indexer.build()
        return

    if args.query:
        print(f"🔍 Searching for: [bold]{args.query}[/bold]")
        retriever = GraphRetriever()
        retriever.set_reranking(not args.no_rerank)
        try:
            response = retriever.query(args.query, top_k=args.top_k, recall_k=args.recall)
            stats = response["stats"]
            results = response["results"]
            console.print(f"[dim]Recall: {stats['total_candidates']} candidates -> Reranked: {stats['reranked']}[/dim]")
            for i, res in enumerate(results):
                score = res.get("score", 0.0)
                score_fmt = f"{score:.4f}"
                color = "green" if i == 0 else "blue"
                title = f"Rank {i+1} | Score: {score_fmt} | {res.get('source_id')}"
                display_text = f"{res.get('context', '')}\n\n{res.get('content', '')}"
                syntax = Syntax(display_text, "python", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=title, border_style=color))
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

    if args.stats:
        import os
        import random
        meta_path = "data/vector_store/metadata.json"
        
        if os.path.exists(meta_path):
             with open(meta_path, 'r') as f:
                 meta = json.load(f)
             
             table = Table(title="RAG Index Statistics")
             table.add_column("Metric", style="cyan")
             table.add_column("Value", style="magenta")
             
             count = meta.get("count", 0)
             table.add_row("Generated At", meta.get("generated_at"))
             table.add_row("Model", meta.get("model"))
             table.add_row("Total Chunks", str(count))
             
             console.print(table)
             
             # Random Sample Chunks
             chunks = meta.get("chunks", [])
             if chunks:
                 sample_size = min(len(chunks), args.num_samples)
                 console.print(f"\n🎲 [bold]Random Sample of {sample_size} Chunks:[/bold]")
                 
                 samples = random.sample(chunks, sample_size)
                 
                 for i, chunk in enumerate(samples):
                     display_text = f"{chunk.get('context', '')}\n\n{chunk.get('content', '')}"
                     syntax = Syntax(display_text, "python", theme="monokai", line_numbers=True)
                     
                     # Enriched Metadata
                     source = chunk.get('source', 'Unknown')
                     lines = chunk.get('lines', '?')
                     nid = chunk.get('id', '?')
                     
                     title = f"Sample {i+1} | Source: {source} | Node: {nid} | Lines: {lines}"
                     console.print(Panel(syntax, title=title, border_style="yellow"))
        else:
             console.print("[red]No index metadata found. Run --build first.[/red]")

if __name__ == "__main__":
    main()
