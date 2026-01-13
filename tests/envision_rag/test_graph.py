"""
Test suite for Graph Tools.
Tests the graph query tools (find_readers, find_writers, describe_impact).
"""
from pathlib import Path

from envision_rag.graph.graph_types import DependencyGraph
from envision_rag.tools.graph_tools import GraphTools


def main():
    """CLI entry point for running graph tool tests."""
    test_tools()


def test_tools():
    """Run basic graph tool tests."""
    # Load graph
    g = DependencyGraph()
    g.load("data/dependency_graph.json")
    print(f"Graph loaded: {g.stats()}")
    
    tools = GraphTools(g)
    
    # Test 1: Readers of Items.ion
    print("\n--- Test 1: Readers of /Clean/Items.ion ---")
    readers = tools.find_readers("/Clean/Items.ion")
    print(f"Found {len(readers)} readers:")
    for r in readers[:5]:  # Limit output
        print(f"  - {r}")
    if len(readers) > 5:
        print(f"  ... and {len(readers) - 5} more")
        
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
    print(impact[:500] + "..." if len(impact) > 500 else impact)
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()

