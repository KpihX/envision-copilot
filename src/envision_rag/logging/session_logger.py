"""
Session Logger for Envision RAG.
Captures all verbose output and persists to JSON for replay.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class LogEvent:
    """Single log event (thought, action, observation, etc.)"""
    event_type: str
    title: str
    content: str
    style: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionLog:
    """Complete session log with metadata and events."""
    session_id: str
    log_type: str  # "main" | "benchmark"
    started_at: str
    ended_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    events: List[LogEvent] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "log_type": self.log_type,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metadata": self.metadata,
            "events": [asdict(e) for e in self.events],
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionLog":
        events = [LogEvent(**e) for e in data.get("events", [])]
        return cls(
            session_id=data["session_id"],
            log_type=data["log_type"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            metadata=data.get("metadata", {}),
            events=events,
            summary=data.get("summary", {})
        )


class SessionLogger:
    """
    Captures session events and persists to JSON.
    
    Usage:
        logger = SessionLogger(log_type="main", log_dir="data/logs")
        logger.start_session({"query": "my question"})
        logger.log_event("thought", "Thinking", "I need to...", "thought")
        logger.end_session({"success": True})
        path = logger.save()
    """
    
    def __init__(self, log_type: str, log_dir: str = "data/logs", enabled: bool = True):
        """
        Initialize logger.
        
        Args:
            log_type: "main" | "benchmark"
            log_dir: Base directory for logs
            enabled: If False, logging is a no-op
        """
        self.log_type = log_type
        self.log_dir = Path(log_dir) / log_type
        self.enabled = enabled
        self.session: Optional[SessionLog] = None
        
        # Ensure directory exists
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new logging session. Returns session ID."""
        if not self.enabled:
            return ""
            
        session_id = str(uuid.uuid4())[:8]
        self.session = SessionLog(
            session_id=session_id,
            log_type=self.log_type,
            started_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        return session_id
    
    def log_event(self, event_type: str, title: str, content: str, style: str = "info"):
        """Log a single event."""
        if not self.enabled or not self.session:
            return
            
        event = LogEvent(
            event_type=event_type,
            title=title,
            content=content,
            style=style
        )
        self.session.events.append(event)
    
    def end_session(self, summary: Optional[Dict[str, Any]] = None):
        """Mark session as ended with optional summary."""
        if not self.enabled or not self.session:
            return
            
        self.session.ended_at = datetime.now().isoformat()
        self.session.summary = summary or {}
    
    def save(self) -> Optional[Path]:
        """Save session to JSON file. Returns path to saved file."""
        if not self.enabled or not self.session:
            return None
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_{self.session.session_id}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.session.to_dict(), f, ensure_ascii=False, indent=2)
        
        return filepath
    
    @classmethod
    def load_nth(cls, log_type: str, nth: int = 1, log_dir: str = "data/logs") -> Optional[SessionLog]:
        """
        Load the nth most recent log.
        
        Args:
            log_type: "main" | "benchmark"
            nth: 1 = most recent, 2 = second most recent, etc.
            log_dir: Base directory for logs
            
        Returns:
            SessionLog or None if not found
        """
        logs = cls.list_logs(log_type, log_dir)
        if not logs or nth > len(logs):
            return None
            
        # Logs are sorted newest first, so nth-1 is the index
        filepath = logs[nth - 1]
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return SessionLog.from_dict(data)
    
    @classmethod
    def list_logs(cls, log_type: str, log_dir: str = "data/logs") -> List[Path]:
        """
        List all logs of a given type, sorted by newest first.
        
        Args:
            log_type: "main" | "benchmark"
            log_dir: Base directory for logs
            
        Returns:
            List of log file paths, newest first
        """
        type_dir = Path(log_dir) / log_type
        if not type_dir.exists():
            return []
            
        logs = sorted(type_dir.glob("*.json"), reverse=True)
        return logs
