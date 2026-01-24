
try:
    from envision_copilot.core.main import EnvisionCopilot
    print("Successfully imported EnvisionCopilot")
    
    agent = EnvisionCopilot(verbose=True)
    print("Successfully instantiated EnvisionCopilot")
    
    # Check if graph is built
    if agent.workflow:
        print("Workflow graph built successfully")
        
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
