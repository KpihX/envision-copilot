# Tool Definitions for LLM Function Calling

TOOLS_SCHEMA = [
    {
        "name": "semantic_search",
        "description": "Searches the codebase for concepts, business logic, or definitions using semantic search (RAG). Use this to find 'Where is X defined?' or 'How does Y work?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query describing what you are looking for."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "structural_explorer",
        "description": "Explores the dependency graph. Use this to find what a file uses (Imports/Reads) or what uses it.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["neighbors", "stats"],
                    "description": "'neighbors': get dependencies. 'stats': get global graph counts."
                },
                "node_id": {
                    "type": "string",
                    "description": "The Node ID (usually Script Path or Variable Name) to explore. Required for 'neighbors' action."
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "read_code_section",
        "description": "Reads a specific section of a file. Use this ONLY when you have a specific file path and line numbers from previous tool outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file."
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line number (1-indexed)."
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line number."
                }
            },
            "required": ["file_path", "start_line", "end_line"]
        }
    },
    {
        "name": "manage_plan",
        "description": "Updates the Tree of Thoughts plan. Use this to add sub-tasks or mark the current task as done/failed.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add_subtask", "mark_done", "mark_failed"],
                    "description": "The action to perform on the current plan node."
                },
                "description": {
                    "type": "string",
                    "description": "Description of the new subtask (required for add_subtask) or reasoning for completion."
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "ask_user",
        "description": "Asks the user for clarification. Use this when the goal is ambiguous or you need human input to proceed.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user."
                }
            },
            "required": ["question"]
        }
    }
]
