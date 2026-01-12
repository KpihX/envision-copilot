import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from envision_rag.graph.graph_types import DependencyGraph
from envision_rag.tools.graph_tools import GraphTools

def test_tools():
    # Load graph
    g = DependencyGraph()
    g.load("data/dependency_graph.json")
    print(f"Graph loaded: {g.stats()}")
    
    tools = GraphTools(g)
    
    # Test 1: Readers of Items.ion
    print("\n--- Test 1: Readers of /Clean/Items.ion ---")
    readers = tools.find_readers("/Clean/Items.ion")
    print(f"Found {len(readers)} readers:")
    for r in readers:
        print(f"  - {r}")
        
    # Test 2: Writers of Items.ion
    print("\n--- Test 2: Writers of /Clean/Items.ion ---")
    writers = tools.find_writers("/Clean/Items.ion")
    print(f"Found {len(writers)} writers:")
    for w in writers:
        print(f"  - {w}")

    # Test 3: Impact Analysis
    script = "/1. utilities/2. preprocess/[ 3 ] - Forecast Autodiff"
    print(f"\n--- Test 3: Impact of {script} ---")
    impact = tools.describe_impact(script)
    print(impact)

if __name__ == "__main__":
    test_tools()
