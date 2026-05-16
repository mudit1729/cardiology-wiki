from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
import re

import markdown
import yaml


WIKILINK_RE = re.compile(r"(!)?\[\[([^\]]+)\]\]")
HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
LOG_ENTRY_RE = re.compile(
    r"^## \[(?P<date>[^\]]+)\] (?P<kind>[^|]+)\| (?P<title>.+)$",
    re.MULTILINE,
)
IGNORE_PARTS = {".git", ".venv", "__pycache__", "static", "templates_html", ".grounding", ".claude", "node_modules"}
ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".pdf"}


@dataclass
class Page:
    path: str
    file_path: Path
    title: str
    raw: str
    html: str
    excerpt: str
    links: list[str]
    backlinks: list[str]
    tags: list[str] = field(default_factory=list)
    status: str = ""
    page_type: str = ""
    year: str = ""
    venue: str = ""
    citations: int = 0
    meta: dict = field(default_factory=dict)


class WikiRepository:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._pages: dict[str, Page] = {}
        self._path_lookup: dict[str, str] = {}
        self._stem_lookup: dict[str, str] = {}
        self._stamp: float | None = None

    def _iter_markdown_files(self) -> Iterable[Path]:
        for file_path in self.base_dir.rglob("*.md"):
            if any(part in IGNORE_PARTS for part in file_path.parts):
                continue
            yield file_path

    def _compute_stamp(self) -> float:
        stamp = 0.0
        for file_path in self._iter_markdown_files():
            stamp = max(stamp, file_path.stat().st_mtime)
        return stamp

    def refresh(self) -> None:
        stamp = self._compute_stamp()
        if self._pages and stamp == self._stamp:
            return

        sources: dict[str, tuple[Path, str, str, dict]] = {}
        stem_index: defaultdict[str, list[str]] = defaultdict(list)

        for file_path in self._iter_markdown_files():
            rel_path = file_path.relative_to(self.base_dir).as_posix()
            page_path = rel_path[:-3]
            raw = file_path.read_text(encoding="utf-8")
            meta = self._extract_frontmatter(raw)
            title = self._extract_title(page_path, raw, meta)
            sources[page_path] = (file_path, raw, title, meta)
            stem_index[Path(page_path).name.lower()].append(page_path)

        self._path_lookup = {path.lower(): path for path in sources}
        self._stem_lookup = {
            stem: matches[0]
            for stem, matches in stem_index.items()
            if len(matches) == 1
        }

        pages: dict[str, Page] = {}
        backlinks: defaultdict[str, list[str]] = defaultdict(list)

        for path, (file_path, raw, title, meta) in sources.items():
            links = self._extract_links(raw)
            for target in links:
                backlinks[target].append(path)

            body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw, count=1, flags=re.DOTALL)
            body_wikilinks = self._render_wikilinks(body)
            body_safe, math_blocks = self._protect_math(body_wikilinks)
            html_raw = markdown.markdown(
                body_safe,
                extensions=[
                    "tables",
                    "fenced_code",
                    "codehilite",
                    "toc",
                    "sane_lists",
                    "attr_list",
                ],
            )
            html = self._restore_math(html_raw, math_blocks)
            excerpt = self._make_excerpt(raw)
            tags = meta.get("tags", []) or []
            if isinstance(tags, str):
                tags = [tags]

            citations_val = meta.get("citations", 0)
            try:
                citations_val = int(citations_val)
            except (ValueError, TypeError):
                citations_val = 0

            pages[path] = Page(
                path=path,
                file_path=file_path,
                title=title,
                raw=raw,
                html=html,
                excerpt=excerpt,
                links=links,
                backlinks=[],
                tags=tags,
                status=str(meta.get("status", "")),
                page_type=str(meta.get("type", "")),
                year=str(meta.get("year", "")),
                venue=str(meta.get("venue", "")),
                citations=citations_val,
                meta=meta,
            )

        for page in pages.values():
            page.backlinks = sorted(
                set(path for path in backlinks.get(page.path, []) if path != page.path)
            )

        self._pages = pages
        self._stamp = stamp

    def list_pages(self) -> list[Page]:
        self.refresh()
        return sorted(self._pages.values(), key=lambda page: page.path)

    def get_page(self, path: str) -> Page | None:
        self.refresh()
        normalized = self._resolve_from_indexes(path)
        if not normalized:
            return None
        return self._pages.get(normalized)

    def resolve_target(self, target: str) -> str | None:
        self.refresh()
        return self._resolve_from_indexes(target)

    def search(self, query: str, limit: int = 25) -> list[Page]:
        self.refresh()
        terms = [term for term in re.split(r"\s+", query.lower()) if term]
        if not terms:
            return []

        scored: list[tuple[int, Page]] = []
        for page in self._pages.values():
            title = page.title.lower()
            body = page.raw.lower()
            tag_str = " ".join(page.tags).lower()
            score = 0
            for term in terms:
                score += title.count(term) * 8
                score += tag_str.count(term) * 4
                score += body.count(term)
            if score:
                scored.append((score, page))

        scored.sort(key=lambda item: (-item[0], item[1].path))
        return [page for _, page in scored[:limit]]

    def grouped_pages(self) -> dict[str, list[Page]]:
        self.refresh()
        groups: defaultdict[str, list[Page]] = defaultdict(list)

        for page in self._pages.values():
            parts = page.path.split("/")
            if parts[0] == "wiki" and len(parts) > 1:
                label = parts[1].replace("-", " ").title()
            elif parts[0] == "raw" and len(parts) > 1:
                label = f"Raw {parts[1].replace('-', ' ').title()}"
            elif parts[0] == "templates":
                label = "Templates"
            else:
                label = "Core"
            groups[label].append(page)

        return {
            label: sorted(items, key=lambda page: page.title.lower())
            for label, items in sorted(groups.items())
        }

    def recent_log_entries(self, limit: int = 8) -> list[dict[str, str]]:
        page = self.get_page("log")
        if not page:
            return []

        entries: list[dict[str, str]] = []
        for match in LOG_ENTRY_RE.finditer(page.raw):
            entries.append(
                {
                    "date": match.group("date").strip(),
                    "kind": match.group("kind").strip(),
                    "title": match.group("title").strip(),
                }
            )
        return list(reversed(entries[-limit:]))

    def featured_pages(self) -> list[Page]:
        featured_paths = [
            "wiki/overview",
            "wiki/taxonomies/research-map",
            "index",
        ]
        featured: list[Page] = []
        for path in featured_paths:
            page = self.get_page(path)
            if page:
                featured.append(page)
        return featured

    def all_tags(self) -> dict[str, int]:
        self.refresh()
        tag_counts: defaultdict[str, int] = defaultdict(int)
        for page in self._pages.values():
            for tag in page.tags:
                tag_counts[tag] += 1
        return dict(sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])))

    def pages_by_tag(self, tag: str) -> list[Page]:
        self.refresh()
        return sorted(
            [p for p in self._pages.values() if tag in p.tags],
            key=lambda p: p.title.lower(),
        )

    def papers(self) -> list[Page]:
        self.refresh()
        return sorted(
            [p for p in self._pages.values() if p.page_type in ("trial", "guideline", "procedure", "conference", "paper", "source-summary")],
            key=lambda p: (p.year or "0000", p.title.lower()),
            reverse=True,
        )

    def paper_collections(self) -> dict[str, list[Page]]:
        all_papers = self.papers()
        collections: dict[str, list[Page]] = {
            "Stable CAD": [],
            "ACS / STEMI": [],
            "Imaging-Guided PCI": [],
            "Physiology-Guided PCI": [],
            "Antiplatelets / DAPT": [],
            "AF + PCI": [],
            "TAVR / Structural Heart": [],
            "Lipid / Prevention": [],
            "Heart Failure": [],
            "Procedures": [],
            "Guidelines": [],
            "Conferences": [],
        }
        for p in all_papers:
            if any(t in p.tags for t in ["stable-cad", "chronic-coronary-disease"]):
                collections["Stable CAD"].append(p)
            if any(t in p.tags for t in ["acs", "stemi", "nstemi", "acute-coronary-syndrome"]):
                collections["ACS / STEMI"].append(p)
            if any(t in p.tags for t in ["ivus", "oct", "imaging-guided-pci", "intravascular-imaging"]):
                collections["Imaging-Guided PCI"].append(p)
            if any(t in p.tags for t in ["ffr", "ifr", "physiology-guided-pci", "coronary-physiology"]):
                collections["Physiology-Guided PCI"].append(p)
            if any(t in p.tags for t in ["antiplatelet", "dapt", "dual-antiplatelet", "antithrombotic"]):
                collections["Antiplatelets / DAPT"].append(p)
            if any(t in p.tags for t in ["atrial-fibrillation", "af-pci", "anticoagulation"]):
                collections["AF + PCI"].append(p)
            if any(t in p.tags for t in ["tavr", "structural-heart", "aortic-stenosis", "tavi"]):
                collections["TAVR / Structural Heart"].append(p)
            if any(t in p.tags for t in ["lipid", "pcsk9", "statin", "prevention", "glp1"]):
                collections["Lipid / Prevention"].append(p)
            if any(t in p.tags for t in ["heart-failure", "sglt2", "hfref", "hfpef"]):
                collections["Heart Failure"].append(p)
            if p.page_type == "procedure":
                collections["Procedures"].append(p)
            if p.page_type == "guideline":
                collections["Guidelines"].append(p)
            if p.page_type == "conference":
                collections["Conferences"].append(p)
        # Fallback: any paper not yet bucketed goes into a "Source-Grounded Papers" collection
        bucketed = {p.path for v in collections.values() for p in v}
        unbucketed = [p for p in all_papers if p.page_type == "paper" and p.path not in bucketed]
        if unbucketed:
            collections["Source-Grounded Papers"] = unbucketed
        return {k: v for k, v in collections.items() if v}

    def stats(self) -> dict[str, object]:
        self.refresh()
        total = len(self._pages)
        trials = len([p for p in self._pages.values() if p.page_type == "trial"])
        guidelines = len([p for p in self._pages.values() if p.page_type == "guideline"])
        procedures = len([p for p in self._pages.values() if p.page_type == "procedure"])
        conferences = len([p for p in self._pages.values() if p.page_type == "conference"])
        concepts = len([p for p in self._pages.values() if p.page_type == "concept"])
        papers = len([p for p in self._pages.values() if p.page_type == "paper"])
        tags = len(self.all_tags())
        return {
            "total": total,
            "trials": trials,
            "guidelines": guidelines,
            "procedures": procedures,
            "conferences": conferences,
            "concepts": concepts,
            "papers": papers,
            "tags": tags,
        }

    def paper_graph_data(self) -> dict:
        self.refresh()
        papers = {p.path: p for p in self._pages.values()
                  if p.page_type in ("trial", "guideline", "procedure", "conference", "paper", "source-summary")}

        def get_group(p: Page) -> str:
            if any(t in p.tags for t in ["stable-cad", "chronic-coronary-disease"]):
                return "stable-cad"
            if any(t in p.tags for t in ["acs", "stemi", "nstemi"]):
                return "acs"
            if any(t in p.tags for t in ["ivus", "oct", "imaging-guided-pci"]):
                return "imaging"
            if any(t in p.tags for t in ["ffr", "ifr", "physiology-guided-pci"]):
                return "physiology"
            if any(t in p.tags for t in ["antiplatelet", "dapt"]):
                return "antiplatelet"
            if any(t in p.tags for t in ["atrial-fibrillation", "af-pci"]):
                return "af-pci"
            if any(t in p.tags for t in ["tavr", "structural-heart"]):
                return "structural"
            if any(t in p.tags for t in ["lipid", "pcsk9", "statin", "prevention", "glp1"]):
                return "prevention"
            if any(t in p.tags for t in ["heart-failure", "sglt2"]):
                return "heart-failure"
            if p.page_type == "procedure":
                return "procedure"
            if p.page_type == "guideline":
                return "guideline"
            if p.page_type == "conference":
                return "conference"
            return "other"

        nodes = []
        for path, p in papers.items():
            nodes.append({
                "id": path,
                "title": p.title,
                "year": p.year,
                "group": get_group(p),
                "citations": p.citations,
                "tags": p.tags,
                "url": f"/page/{path}",
            })

        node_ids = set(papers.keys())
        edges = []
        seen = set()
        for path, p in papers.items():
            for link in p.links:
                if link in node_ids and link != path:
                    edge_key = tuple(sorted([path, link]))
                    if edge_key not in seen:
                        seen.add(edge_key)
                        edges.append({"source": path, "target": link})
            for bl in p.backlinks:
                if bl in node_ids and bl != path:
                    edge_key = tuple(sorted([path, bl]))
                    if edge_key not in seen:
                        seen.add(edge_key)
                        edges.append({"source": bl, "target": path})

        return {"nodes": nodes, "edges": edges}

    def timeline_data(self) -> dict[str, list[dict]]:
        self.refresh()
        directions = {
            "Stable CAD": lambda p: any(t in p.tags for t in ["stable-cad", "chronic-coronary-disease"]),
            "ACS / STEMI": lambda p: any(t in p.tags for t in ["acs", "stemi", "nstemi"]),
            "Imaging-Guided PCI": lambda p: any(t in p.tags for t in ["ivus", "oct", "imaging-guided-pci"]),
            "Physiology-Guided PCI": lambda p: any(t in p.tags for t in ["ffr", "ifr", "physiology-guided-pci"]),
            "Antiplatelets": lambda p: any(t in p.tags for t in ["antiplatelet", "dapt"]),
            "AF + PCI": lambda p: any(t in p.tags for t in ["atrial-fibrillation", "af-pci"]),
            "TAVR / Structural": lambda p: any(t in p.tags for t in ["tavr", "structural-heart"]),
            "Lipid / Prevention": lambda p: any(t in p.tags for t in ["lipid", "pcsk9", "statin", "prevention", "glp1"]),
            "Heart Failure": lambda p: any(t in p.tags for t in ["heart-failure", "sglt2"]),
        }

        result: dict[str, list[dict]] = {}
        for direction, pred in directions.items():
            papers = [p for p in self._pages.values()
                      if p.page_type in ("trial", "paper", "source-summary") and pred(p)]
            papers.sort(key=lambda p: (p.year or "0000", p.title))
            result[direction] = [{
                "path": p.path,
                "title": p.title,
                "year": p.year,
                "citations": p.citations,
                "url": f"/page/{p.path}",
            } for p in papers]

        return {k: v for k, v in result.items() if v}

    def _extract_frontmatter(self, raw: str) -> dict:
        match = FRONTMATTER_RE.match(raw)
        if not match:
            return {}
        fm_text = match.group(1)
        try:
            return yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            fixed_lines = []
            for line in fm_text.split("\n"):
                stripped = line.lstrip()
                if stripped.startswith("-") or stripped.startswith("#") or not stripped:
                    fixed_lines.append(line)
                    continue
                colon_pos = stripped.find(":")
                if colon_pos > 0:
                    key = stripped[:colon_pos]
                    val = stripped[colon_pos + 1:].strip()
                    if val and ":" in val and not val.startswith('"') and not val.startswith("'") and not val.startswith("["):
                        indent = line[:len(line) - len(stripped)]
                        val_escaped = val.replace('"', '\\"')
                        line = f'{indent}{key}: "{val_escaped}"'
                fixed_lines.append(line)
            try:
                return yaml.safe_load("\n".join(fixed_lines)) or {}
            except yaml.YAMLError:
                return {}

    def _extract_title(self, page_path: str, raw: str, meta: dict | None = None) -> str:
        if meta:
            fm_title = meta.get("title")
            if isinstance(fm_title, str) and fm_title.strip():
                return fm_title.strip()
        body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw, count=1, flags=re.DOTALL)
        body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
        body = re.sub(r"~~~.*?~~~", "", body, flags=re.DOTALL)
        match = HEADING_RE.search(body)
        if match:
            return match.group(1).strip()
        return Path(page_path).name.replace("-", " ").title()

    def _make_excerpt(self, raw: str, max_length: int = 220) -> str:
        text = re.sub(r"(?m)^---.*?^---\s*", "", raw, flags=re.DOTALL)
        text = re.sub(r"(?m)^#+\s.*$", "", text)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
        text = re.sub(r"\$[^$\n]+\$", " ", text)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", lambda m: m.group(1).split("|")[-1], text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?!\w)", r"\1", text)
        text = re.sub(r"(?<![\w_])_([^_\n]+)_(?!\w)", r"\1", text)
        text = re.sub(r"`([^`\n]+)`", r"\1", text)
        text = re.sub(r"(?m)^[\s>]*[-*+]\s+", "", text)
        text = re.sub(r"(?m)^>\s*", "", text)
        text = re.sub(r"(?m)^---+\s*$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_length:
            return text
        return text[: max_length - 1].rstrip() + "…"

    @staticmethod
    def _protect_math(text: str) -> tuple[str, list[tuple[str, str]]]:
        math_blocks: list[tuple[str, str]] = []

        def _stash_display(m: re.Match) -> str:
            idx = len(math_blocks)
            math_blocks.append(("display", m.group(0)))
            return f"\n\n@@WIKI_MATH_{idx}@@\n\n"

        def _stash_inline(m: re.Match) -> str:
            idx = len(math_blocks)
            math_blocks.append(("inline", m.group(0)))
            return f"@@WIKI_MATH_{idx}@@"

        text = re.sub(r"\$\$(.+?)\$\$", _stash_display, text, flags=re.DOTALL)
        text = re.sub(r"(?<!\$)\$(?!\$)(.+?)\$(?!\$)", _stash_inline, text)
        return text, math_blocks

    @staticmethod
    def _restore_math(html: str, math_blocks: list[tuple[str, str]]) -> str:
        for i, (kind, original) in enumerate(math_blocks):
            tag = f"@@WIKI_MATH_{i}@@"
            html = html.replace(f"<p>{tag}</p>", original)
            html = html.replace(tag, original)
        return html

    def _extract_links(self, raw: str) -> list[str]:
        targets: list[str] = []
        for _, payload in WIKILINK_RE.findall(raw):
            target = payload.split("|", 1)[0].split("#", 1)[0].strip()
            resolved = self._resolve_from_indexes(target)
            if resolved:
                targets.append(resolved)
        return sorted(set(targets))

    def _render_wikilinks(self, raw: str) -> str:
        def replace(match: re.Match[str]) -> str:
            is_embed = bool(match.group(1))
            payload = match.group(2).strip()

            if "|" in payload:
                target_part, label = payload.split("|", 1)
                label = label.strip()
            else:
                target_part, label = payload, ""

            if "#" in target_part:
                target, anchor = target_part.split("#", 1)
                anchor_slug = "#" + self._slugify(anchor)
            else:
                target, anchor_slug = target_part, ""

            target = target.strip()
            resolved = self._resolve_from_indexes(target)

            if not resolved and target.startswith("."):
                return label or Path(target).name.replace("-", " ").replace("_", " ").title()
            resolved = resolved or target.strip().lstrip("/")

            if is_embed:
                asset_path = resolved if any(resolved.endswith(suffix) for suffix in ASSET_SUFFIXES) else target
                src = f"/vault/{quote(asset_path)}"
                alt = label or Path(asset_path).name
                return f'<img class="embedded-asset" src="{src}" alt="{alt}">'

            href = f"/page/{quote(resolved)}{anchor_slug}"
            text = label or Path(resolved).name.replace("-", " ").replace("_", " ").title()
            return f"[{text}]({href})"

        return WIKILINK_RE.sub(replace, raw)

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^\w\s-]", "", value.lower()).strip()
        return re.sub(r"[-\s]+", "-", slug)

    def _resolve_from_indexes(self, target: str) -> str | None:
        cleaned = target.strip().lstrip("/")
        if cleaned.endswith(".md"):
            cleaned = cleaned[:-3]
        exact = self._path_lookup.get(cleaned.lower())
        if exact:
            return exact
        return self._stem_lookup.get(Path(cleaned).name.lower())
