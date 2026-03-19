from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_agent.web import _render_app_html


def main() -> None:
    root = ROOT
    html = _render_app_html()
    public_dir = root / "frontend" / "public"
    dist_dir = root / "frontend" / "dist"
    public_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "legacy.html").write_text(html, encoding="utf-8")
    (dist_dir / "legacy.html").write_text(html, encoding="utf-8")
    (dist_dir / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
