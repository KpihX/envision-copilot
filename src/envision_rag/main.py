import argparse
import sys
import yaml
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).parent.parent))

from envision_rag.graph.builder import GraphBuilder
from envision_rag.tools.graph_tools import GraphTools
from envision_rag.workflow.agent import AgentWorkflow

def load_config(path: str = "config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Envision Graph RAG")
    parser.add_argument("--query", "-q", help="Single query mode")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of graph")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full agent trace")
    args = parser.parse_args()

    print("🚀 Envision RAG System (Graph-Enhanced)")
    
    # 1. Config
    config = load_config()
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    graph_path = data_dir / "dependency_graph.json"

    # 2. Graph
    builder = GraphBuilder("./env_scripts")
    graph = None
    
    if args.rebuild or not graph_path.exists():
        print("🏗️ Building Graph...")
        graph = builder.build()
        graph.save(str(graph_path))
    else:
        print("📂 Loading Graph...")
        from envision_rag.graph.graph_types import DependencyGraph
        graph = DependencyGraph()
        graph.load(str(graph_path))
        print(f"   Stats: {graph.stats()}")

    # 3. Agent
    tools = GraphTools(graph)
    workflow = AgentWorkflow(config, tools, verbose=args.verbose)
    app = workflow.build_graph()

    # 4. Interaction
    if args.query:
        print(f"\n❓ Query: {args.query}")
        result = app.invoke({"question": args.query, "scratchpad": "", "messages": [], "facts": []})
        
        # Trace is now live-streamed if verbose
        # if args.verbose: ... 

        print("\n" + "="*40)
        messages = result.get('messages', [])
        
        # Only print final answer if NOT verbose (verbose already showed it Live)
        # Or always print it cleanly at the end?
        # User wants "Unified". Live should be enough.
        # But for non-verbose, we need output.
        
        last_msg = messages[-1] if messages else "No response."
        if not args.verbose:
             if "Final Answer:" in last_msg:
                 print(f"\n🤖 {last_msg.split('Final Answer:')[-1].strip()}")
             else:
                 print(f"\n🤖 {last_msg}")

    elif args.interactive:
        print("\n💬 Interactive Mode (Ctrl+C to exit)")
        while True:
            try:
                q = input("\n>> ")
                if not q.strip(): continue
                q = input("\n>> ")
                if not q.strip(): continue
                result = app.invoke({"question": q, "scratchpad": "", "messages": [], "facts": []})
                
                print("\n🤖 Agent:")
                messages = result.get('messages', [])
                if messages:
                    # Print the last message (Final Answer or last thought)
                    print(messages[-1])
                    # Optionally print the full scratchpad for transparency?
                    # print("\n(Trace):\n" + result['scratchpad'])
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
