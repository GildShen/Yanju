from __future__ import annotations

from typing import Any


def _stringify_candidate(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _stringify_candidate(item)
            if text:
                return text
        return ""
    for attr in ("get", "extract", "text", "html", "body", "content"):
        candidate = getattr(value, attr, None)
        if candidate is None:
            continue
        try:
            candidate_value = candidate() if callable(candidate) else candidate
        except Exception:
            continue
        text = _stringify_candidate(candidate_value)
        if text:
            return text
    return str(value)



def _looks_like_markup(value: str) -> bool:
    sample = value[:500].lower()
    return "<html" in sample or "<body" in sample or "<meta" in sample or "<div" in sample or "<section" in sample



def _page_markup(page: Any) -> str:
    for attr in ("html", "body", "content", "source", "markup"):
        candidate = getattr(page, attr, None)
        if candidate is None:
            continue
        try:
            value = candidate() if callable(candidate) else candidate
        except Exception:
            continue
        text = _stringify_candidate(value)
        if text and _looks_like_markup(text):
            return text
    rendered = str(page)
    return rendered if _looks_like_markup(rendered) else ""



def fetch_page_html_with_scrapling(url: str) -> str:
    try:
        from scrapling.fetchers import DynamicFetcher, Fetcher
    except ImportError:
        return ""

    fetch_attempts: list[tuple[Any, dict[str, Any]]] = [
        (
            Fetcher,
            {
                "stealthy_headers": True,
                "impersonate": "chrome",
                "follow_redirects": True,
                "timeout": 30,
            },
        ),
    ]

    if any(domain in url for domain in ("pubsonline.informs.org", "onlinelibrary.wiley.com", "linkinghub.elsevier.com", "sciencedirect.com")):
        fetch_attempts.append(
            (
                DynamicFetcher,
                {
                    "headless": True,
                    "network_idle": True,
                    "disable_resources": False,
                    "timeout": 45,
                },
            )
        )

    for fetcher, kwargs in fetch_attempts:
        fetch = getattr(fetcher, "fetch", None) or getattr(fetcher, "get", None)
        if fetch is None:
            continue
        try:
            page = fetch(url, **kwargs)
        except Exception:
            continue
        html = _page_markup(page)
        if html:
            return html
    return ""
