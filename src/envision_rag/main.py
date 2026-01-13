import argparse
import sys
import yaml
from pathlib import Path

# Fix imports
sys.path.append(str(Path(__file__).parent.parent))

from envision_rag.graph.builder import GraphBuilder
from envision_rag.tools.graph_tools import GraphTools
from envision_rag.workflow.agent import AgentWorkflow
from envision_rag.logging.session_logger import SessionLogger

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
    
    # Initialize logger from config
    logging_config = config.get("logging", {})
    logger = SessionLogger(
        log_type="main",
        log_dir=logging_config.get("log_dir", "data/logs"),
        enabled=logging_config.get("enabled", True)
    )

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

    # 3. Agent (with logger)
    tools = GraphTools(graph)
    workflow = AgentWorkflow(config, tools, verbose=args.verbose, logger=logger)
    app = workflow.build_graph()

    # 4. Interaction
    if args.query:
        # Start logging session
        logger.start_session({"query": args.query, "mode": "single", "verbose": args.verbose})
        
        print(f"\n❓ Query: {args.query}")
        result = app.invoke({"question": args.query, "scratchpad": "", "messages": [], "facts": []})

        print("\n" + "="*40)
        messages = result.get('messages', [])
        last_msg = messages[-1] if messages else "No response."
        
        if not args.verbose:
             if "Final Answer:" in last_msg:
                 print(f"\n🤖 {last_msg.split('Final Answer:')[-1].strip()}")
             else:
                 print(f"\n🤖 {last_msg}")
        
        # End and save session
        logger.end_session({"final_answer": last_msg[:500], "success": True})
        log_path = logger.save()
        if log_path and args.verbose:
            print(f"\n📝 Session saved: {log_path}")

    elif args.interactive:
        print("\n💬 Interactive Mode (Ctrl+C to exit)")
        session_count = 0
        while True:
            try:
                q = input("\n>> ")
                if not q.strip(): continue
                
                # Start new session for each question
                session_count += 1
                logger.start_session({"query": q, "mode": "interactive", "session_num": session_count})
                
                result = app.invoke({"question": q, "scratchpad": "", "messages": [], "facts": []})
                
                print("\n🤖 Agent:")
                messages = result.get('messages', [])
                if messages:
                    print(messages[-1])
                
                # Save session
                logger.end_session({"final_answer": messages[-1][:500] if messages else "", "success": True})
                logger.save()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

