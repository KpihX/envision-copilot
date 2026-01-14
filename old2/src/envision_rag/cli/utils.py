import sys
from pathlib import Path
from rich.console import Console

from envision_rag.cli.build_index import build_graph, build_vector
from envision_rag.config_manager import get_config

console = Console()

def ensure_indexes(config: dict):
    """
    Ensure Graph and Vector indexes exist. 
    If not, automatically build them.
    """
    system_conf = config.get("system", {})
    index_conf = config.get("index", {})
    
    data_dir = Path(system_conf.get("data_dir", "./data"))
    store_path = Path(index_conf.get("store_path", "data/vector_store"))
    
    graph_path = data_dir / "dependency_graph.json"
    index_path = store_path / "faiss.index"
    
    missing_graph = not graph_path.exists()
    missing_vector = not index_path.exists()
    
    if missing_graph or missing_vector:
        console.print("\n[yellow bold]⚠️ Indexes missing. Auto-building...[/yellow bold]")
        
        if missing_graph:
            console.print("   [dim]Building Dependency Graph...[/dim]")
            build_graph(config, verbose=False)
            
        if missing_vector:
            console.print("   [dim]Building Vector Index...[/dim]")
            build_vector(config, verbose=False)
            
        console.print("[green]✅ Indexes ready. Proceeding...[/green]\n")
