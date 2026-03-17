from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlencode

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


HUGGINGFACE_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"
DEFAULT_USER_AGENT = "ResearchAgent/1.0 (+https://huggingface.co/docs/hub/api)"


@dataclass
class HuggingFacePaperRecord:
    title: str
    authors: list[str]
    published: str
    abstract: str
    link: str
    source: str
    doi: str = ""
    pdf_url: str = ""
    tags: list[str] | None = None


class HuggingFaceDailyPapersClient:
    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    )
    def fetch_daily_papers(self, date: str | None = None, limit: int = 100) -> list[HuggingFacePaperRecord]:
        params: dict[str, Any] = {"limit": limit}
        if date:
            params["date"] = date
        url = HUGGINGFACE_DAILY_PAPERS_URL
        if params:
            url = f"{url}?{urlencode(params)}"
            
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"})
            response.raise_for_status()
            payload = response.json()
            
        papers = self._extract_papers(payload)
        normalized: list[HuggingFacePaperRecord] = []
        for item in papers:
            parsed = self._parse_record(item, fallback_date=date)
            if parsed is not None:
                normalized.append(parsed)
        return normalized

    def _extract_papers(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("papers", "items", "dailyPapers", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def _parse_record(self, item: dict[str, Any], fallback_date: str | None) -> HuggingFacePaperRecord | None:
        paper_data = item.get("paper") if isinstance(item.get("paper"), dict) else {}
        title = self._clean_text(item.get("title") or paper_data.get("title"))
        if not title:
            return None
        authors = self._parse_authors(item.get("authors") or paper_data.get("authors"))
        abstract = self._clean_text(item.get("summary") or item.get("abstract") or paper_data.get("summary") or paper_data.get("abstract"))
        doi = self._clean_text(item.get("doi") or paper_data.get("doi"))
        arxiv_id = self._clean_text(item.get("id") or item.get("arxiv_id") or paper_data.get("id") or paper_data.get("arxiv_id"))
        published = self._normalize_published(item.get("publishedAt") or item.get("published_at") or item.get("published") or paper_data.get("publishedAt") or paper_data.get("published_at") or paper_data.get("published") or fallback_date)
        link = self._clean_text(item.get("url") or item.get("paper_url") or item.get("paperUrl") or paper_data.get("url") or paper_data.get("paper_url") or paper_data.get("paperUrl"))
        pdf_url = self._clean_text(item.get("pdf_url") or item.get("pdfUrl") or paper_data.get("pdf_url") or paper_data.get("pdfUrl"))
        if not link and arxiv_id:
            link = f"https://huggingface.co/papers/{arxiv_id}"
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return HuggingFacePaperRecord(
            title=title,
            authors=authors,
            published=published,
            abstract=abstract,
            link=link,
            source="Hugging Face Daily Papers",
            doi=doi,
            pdf_url=pdf_url,
            tags=["huggingface-daily-papers"],
        )

    def _parse_authors(self, raw_authors: Any) -> list[str]:
        if isinstance(raw_authors, str):
            return [name.strip() for name in raw_authors.split(",") if name.strip()]
        if isinstance(raw_authors, list):
            authors: list[str] = []
            for item in raw_authors:
                if isinstance(item, str) and item.strip():
                    authors.append(item.strip())
                elif isinstance(item, dict):
                    name = self._clean_text(item.get("name") or item.get("fullname") or item.get("full_name"))
                    if name:
                        authors.append(name)
            return authors
        return []

    def _normalize_published(self, value: Any) -> str:
        raw = self._clean_text(value)
        if not raw:
            return datetime.now(UTC).date().isoformat()
        if len(raw) >= 10:
            prefix = raw[:10]
            try:
                return date.fromisoformat(prefix).isoformat()
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return datetime.now(UTC).date().isoformat()

    def _clean_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).strip()
