from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    feeds_path: Path
    data_dir: Path
    vault_dir: Path

    @property
    def dois_path(self) -> Path:
        return Path("dois.txt")

    @property
    def default_pdf_dir(self) -> Path:
        return Path("papers") / "tmp"

    @property
    def done_pdf_dir(self) -> Path:
        return Path("papers") / "done"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "research_agent.db"

    @property
    def legacy_seen_path(self) -> Path:
        return self.data_dir / "seen.json"

    @property
    def legacy_library_path(self) -> Path:
        return self.data_dir / "library.json"

    @property
    def literature_dir(self) -> Path:
        return self.vault_dir / "literature"

    @property
    def weekly_notes_dir(self) -> Path:
        return self.vault_dir / "weekly-notes"

    @property
    def analysis_notes_dir(self) -> Path:
        return self.vault_dir / "analysis-notes"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.literature_dir.mkdir(parents=True, exist_ok=True)
        self.weekly_notes_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_notes_dir.mkdir(parents=True, exist_ok=True)
        self.done_pdf_dir.mkdir(parents=True, exist_ok=True)
