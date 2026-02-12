from typing import TypedDict, List, Dict, Optional, Any

class CopilotState(TypedDict):
    """
    Global State for the Envision Copilot Graph.
    Includes explicit language tracking and workflow control.
    """
    # 1. Input Context
    original_question: str      # User's raw input (source language)
    user_language: str          # Detected language (e.g., "French")
    
    # 2. Working Context (English)
    question: str               # Translated technical question in English
    
    # 3. Execution State
    current_node_id: Optional[str]
    last_layer_results: List[Dict]  # Results from tools in the last layer
    plan_thought: Optional[str]     # The final reasoning/thought logic from the Thinker phase
    
    # 4. Control Flow
    should_stop: bool           # Signal to stop thinking/acting
    stop_reason: str            # "max_depth", "llm_decision", "irrelevant"
    
    # 5. Output
    final_answer: str           # Final response in user_language

    # 6. Interactive Mode
    interactive_mode: bool      # Whether the copilot is in interactive mode
    
    # 7. Depth Tracking (Live Mode)
    turn_start_depth: int       # Depth at the start of the current interaction
    max_depth: int              # Maximum depth for this session
    
    # 8. Persistence
    last_thought_process: Optional[str] # The thought process from the previous Think step
