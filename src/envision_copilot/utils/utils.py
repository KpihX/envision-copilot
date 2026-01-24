from typing import Any, Dict, List

def smart_truncate(obj: Any, max_lines: int = 100) -> Any:
    """
    Recursively truncates long strings/lists in dictionaries or lists.
    Strategy: Keep first n/2 lines + last n/2 lines, mask middle.
    """
    if isinstance(obj, str):
        lines = obj.split('\n')
        if len(lines) > max_lines:
            half = max_lines // 2
            masked_count = len(lines) - max_lines
            first_part = '\n'.join(lines[:half])
            last_part = '\n'.join(lines[-half:])
            return f"{first_part}\n... [{masked_count} lignes masquées] ...\n{last_part}"
        return obj
    elif isinstance(obj, dict):
        return {k: smart_truncate(v, max_lines) for k, v in obj.items()}
    elif isinstance(obj, list):
        if len(obj) > max_lines:
            half = max_lines // 2
            masked_count = len(obj) - max_lines
            return obj[:half] + [f"... [{masked_count} éléments masqués] ..."] + obj[-half:]
        return [smart_truncate(i, max_lines) for i in obj]
    return obj
