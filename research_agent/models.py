from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PaperEntry:
    entry_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    link: str
    source: str
    note_path: str
    added_at: str
    doi: str = ""
    pdf_url: str = ""
    ai_summary: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
