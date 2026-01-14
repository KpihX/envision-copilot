import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from code_rag.indexer import Indexer
from code_rag.retriever import Retriever

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Code RAG CLI")
    parser.add_argument("--build", action="store_true", help="Build the vector index.")
    parser.add_argument("--query", type=str, help="Semantic search query.")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results.")
    
    args = parser.parse_args()
    
    if args.build:
        try:
            indexer = Indexer()
            indexer.build()
        except Exception as e:
            console.print(f"[red]Build Failed:[/red] {e}")
            sys.exit(1)
            
    if args.query:
        try:
            retriever = Retriever()
            results = retriever.retrieve(args.query, k=args.top_k*3, rerank_top_k=args.top_k)
            
            console.print(Panel(f"[bold cyan]Query:[/bold cyan] {args.query}", border_style="cyan"))
            
            for i, res in enumerate(results, 1):
                header = f"[bold]#{i}[/bold] Path: {res['source_id']} (Score: {res['rerank_score']:.4f})"
                code = res['text']
                # Limit display length
                if len(code) > 1000:
                    code = code[:1000] + "\n... (truncated)"
                    
                console.print(Panel(
                    Syntax(code, "python", theme="monokai", word_wrap=True),
                    title=header,
                    border_style="green"
                ))
                
        except Exception as e:
            console.print(f"[red]Query Failed:[/red] {e}")
    
    if not (args.build or args.query):
        parser.print_help()

if __name__ == "__main__":
    main()
