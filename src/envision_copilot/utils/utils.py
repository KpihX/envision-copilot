from typing import Any, Dict, List

def smart_truncate(obj: Any, max_lines: int = 20, max_items: int = 20) -> Any:
    """
    Recursively truncates long strings/lists in dictionaries or lists.
    Strategy: Keep first n/2 items + last n/2 items, mask middle.
    
    Args:
        obj: The object to truncate.
        max_lines: Max lines for strings.
        max_items: Max items for lists.
    """
    if isinstance(obj, str):
        lines = obj.split('\n')
        if len(lines) > max_lines:
            half = max_lines // 2
            masked_count = len(lines) - max_lines
            first_part = '\n'.join(lines[:half])
            last_part = '\n'.join(lines[-half:])
            return f"{first_part}\n... [{masked_count} masked lines] ...\n{last_part}"
        return obj
    elif isinstance(obj, dict):
        return {k: smart_truncate(v, max_lines, max_items) for k, v in obj.items()}
    elif isinstance(obj, list):
        if len(obj) > max_items:
            masked_count = len(obj) - max_items
            return [smart_truncate(i, max_lines, max_items) for i in obj[:max_items]] + \
                   [f"... [{masked_count} masked items] ..."]
        return [smart_truncate(i, max_lines, max_items) for i in obj]
    return obj
