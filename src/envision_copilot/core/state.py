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
    
    # 4. Control Flow
    should_stop: bool           # Signal to stop thinking/acting
    stop_reason: str            # "max_depth", "llm_decision", "irrelevant"
    
    # 5. Output
    final_answer: str           # Final response in user_language
