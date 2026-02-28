from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    id: str
    filename: str
    filepath: str
    status: str = "uploading"
    markdown_path: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[str] = None
    page_count: Optional[int] = None


@dataclass
class Conversation:
    id: str
    paper_id: str
    title: Optional[str] = None


@dataclass
class Message:
    id: str
    conversation_id: str
    role: str
    content: str


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)