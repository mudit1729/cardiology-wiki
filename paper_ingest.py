"""Paper ingestion pipeline for the Cardio Wiki.

Stream events as we move through:
  1. Detect / acquire PDF (URL or upload)
  2. Mistral OCR  → ground-truth markdown
  3. Semantic Scholar lookup (with XAI fallback) → metadata + citations
  4. DeepSeek summarization → structured wiki draft
  5. OpenAI GPT-5.5 final integration → polished wiki page (with Claude/DeepSeek fallbacks)
  6. Write wiki/sources/papers/{slug}.md + .grounding/md_fc/{slug}.md
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Generator, Tuple

import requests


BASE_DIR = Path(__file__).resolve().parent
GROUNDING_DIR = BASE_DIR / ".grounding" / "md_fc"
PAPERS_DIR = BASE_DIR / "wiki" / "sources" / "papers"


# ---------------------------------------------------------------------------
# 1. Source detection
# ---------------------------------------------------------------------------

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([\d.]+)(?:v\d+)?", re.I)
BIORXIV_RE = re.compile(r"biorxiv\.org/content/([\d./v]+)", re.I)
ELIFE_RE = re.compile(r"elifesciences\.org/articles/(\d+)", re.I)
DOI_RE = re.compile(r"(?:doi\.org/|^)(10\.\d{4,9}/[^\s]+)", re.I)
PMC_RE = re.compile(r"PMC(\d+)", re.I)
PUBMED_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.I)


def detect_source(url: str) -> dict:
    url = url.strip()
    if not url:
        raise ValueError("URL is empty")

    if m := ARXIV_RE.search(url):
        arxiv_id = m.group(1)
        return {
            "kind": "arxiv",
            "id": arxiv_id,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        }
    if m := ELIFE_RE.search(url):
        elife_id = m.group(1)
        return {
            "kind": "elife",
            "id": elife_id,
            "pdf_url": f"https://elifesciences.org/articles/{elife_id}.pdf",
            "abs_url": f"https://elifesciences.org/articles/{elife_id}",
        }
    if m := BIORXIV_RE.search(url):
        biorxiv_id = m.group(1).rstrip("/")
        return {
            "kind": "biorxiv",
            "id": biorxiv_id,
            "pdf_url": f"https://www.biorxiv.org/content/{biorxiv_id}.full.pdf",
            "abs_url": f"https://www.biorxiv.org/content/{biorxiv_id}",
        }
    if "ploscompbiol" in url or "plosbiology" in url or "ploscb" in url.lower():
        m = DOI_RE.search(url)
        if m:
            doi = m.group(1)
            return {
                "kind": "plos",
                "id": doi,
                "pdf_url": f"https://journals.plos.org/ploscompbiol/article/file?id={doi}&type=printable",
                "abs_url": url,
            }
    if url.lower().endswith(".pdf"):
        return {"kind": "direct_pdf", "id": None, "pdf_url": url, "abs_url": url}
    if m := DOI_RE.search(url):
        return {"kind": "doi", "id": m.group(1), "pdf_url": url, "abs_url": url}

    return {"kind": "unknown", "id": None, "pdf_url": url, "abs_url": url}


# ---------------------------------------------------------------------------
# 2. Mistral OCR
# ---------------------------------------------------------------------------


def mistral_ocr_url(url: str, timeout: int = 600) -> str:
    key = os.environ["MISTRAL_KEY"]
    payload = {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": url},
    }
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/ocr",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    md = "\n\n".join(p["markdown"] for p in resp.get("pages", []))
    if len(md) < 5000:
        raise RuntimeError(f"OCR returned only {len(md)} chars — likely an HTML stub")
    return md


def mistral_ocr_bytes(pdf_bytes: bytes, filename: str = "paper.pdf", timeout: int = 600) -> str:
    key = os.environ["MISTRAL_KEY"]
    boundary = "----n" + os.urandom(8).hex()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nocr\r\n".encode()
        + f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
          f"Content-Type: application/pdf\r\n\r\n".encode()
        + pdf_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        "https://api.mistral.ai/v1/files",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    file_id = json.loads(urllib.request.urlopen(req, timeout=120).read())["id"]
    req = urllib.request.Request(
        f"https://api.mistral.ai/v1/files/{file_id}/url?expiry=24",
        headers={"Authorization": f"Bearer {key}"},
    )
    signed = json.loads(urllib.request.urlopen(req, timeout=30).read())["url"]
    return mistral_ocr_url(signed, timeout=timeout)


# ---------------------------------------------------------------------------
# 3. Semantic Scholar lookup (with XAI fallback)
# ---------------------------------------------------------------------------


def semantic_scholar_lookup(*, arxiv_id: str | None = None, doi: str | None = None,
                             title: str | None = None) -> dict | None:
    base = "https://api.semanticscholar.org/graph/v1/paper"
    fields = "title,authors,year,venue,citationCount,abstract,externalIds"
    try:
        if arxiv_id:
            url = f"{base}/ArXiv:{arxiv_id}?fields={fields}"
        elif doi:
            url = f"{base}/DOI:{urllib.parse.quote(doi, safe='')}?fields={fields}"
        elif title:
            q = urllib.parse.quote(title)
            url = f"{base}/search?query={q}&limit=1&fields={fields}"
        else:
            return None
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if "data" in data:
            data = data["data"][0] if data["data"] else None
        if not data:
            return None
        return {
            "title": data.get("title"),
            "authors": [a.get("name") for a in (data.get("authors") or [])],
            "year": data.get("year"),
            "venue": data.get("venue"),
            "citations": data.get("citationCount") or 0,
            "abstract": data.get("abstract") or "",
            "external_ids": data.get("externalIds") or {},
        }
    except Exception:
        return None


def xai_citation_estimate(title: str, year: int | None = None) -> int | None:
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        return None
    prompt = (
        f"What is the approximate citation count for the paper titled \"{title}\""
        f"{' (' + str(year) + ')' if year else ''}? "
        "Return ONLY a single integer (the citation count) or the word UNKNOWN. "
        "No other text."
    )
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "grok-3",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
                "temperature": 0,
            },
            timeout=60,
        )
        text = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\d{1,7}", text)
        return int(m.group(0)) if m else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. DeepSeek summarization
# ---------------------------------------------------------------------------


def _strip_frontmatter(md: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", md, count=1, flags=re.DOTALL).strip()


def deepseek_summarize(ocr_md: str, metadata: dict) -> str:
    key = os.environ["DEEPSEEK_API_KEY"]
    title = metadata.get("title", "Unknown title")
    authors = ", ".join(metadata.get("authors") or [])
    year = metadata.get("year") or "unknown"
    venue = metadata.get("venue") or "unknown"

    prompt = (
        f"You are summarizing a research paper for the Cardio Wiki — an interventional "
        f"cardiology intelligence system covering practice-changing trials, guidelines, "
        f"PCI techniques, structural heart interventions, and India-specific practice.\n\n"
        f"Paper: {title}\n"
        f"Authors: {authors}\n"
        f"Year: {year} · Venue: {venue}\n\n"
        f"Produce a Markdown summary with these sections:\n"
        f"- Clinical Question (1 sentence)\n"
        f"- PICO (table: Population, Intervention, Comparator, Outcome)\n"
        f"- Key Results (specific numbers, hazard ratios, p-values)\n"
        f"- What Changed? (practice impact)\n"
        f"- What Did Not Change? (things not proven)\n"
        f"- How I Would Explain This to a Patient (plain language)\n"
        f"- Relevance for Indian Practice (cost, availability, follow-up)\n"
        f"- Connections (use wikilinks like [[../../concepts/foo|Foo]] for these "
        f"concept pages: interventional-cardiology, coronary-artery-disease, "
        f"acute-coronary-syndromes, antiplatelet-therapy, structural-heart, "
        f"heart-failure, india-practice, coronary-physiology)\n\n"
        f"Be clinically precise. Quote exact numbers from the paper. "
        f"Do not invent results. The full OCR follows.\n\n"
        f"--- OCR ---\n{ocr_md[:120000]}"
    )

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("DEEPSEEK_INGEST_MODEL", "deepseek-chat")
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,
            "temperature": 0.2,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 5. XAI Grok final integration (with DeepSeek fallback)
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", (title or "untitled").lower())
    s = re.sub(r"\s+", "-", s).strip("-")
    return (s or "paper")[:80]


def _build_frontmatter(metadata: dict, slug: str, source_info: dict) -> str:
    title = metadata.get("title") or slug
    authors = metadata.get("authors") or []
    year = metadata.get("year") or "null"
    venue = metadata.get("venue") or "null"
    citations = metadata.get("citations") or 0
    arxiv_id = source_info.get("id") if source_info.get("kind") == "arxiv" else None
    doi = (metadata.get("external_ids") or {}).get("DOI")
    today = time.strftime("%Y-%m-%d")

    lines = [
        "---",
        f'title: "{title}"',
        "type: paper",
        "status: active",
        f"updated: {today}",
        f"year: {year}",
    ]
    if venue and venue != "null":
        lines.append(f'venue: "{venue}"')
    if arxiv_id:
        lines.append(f'arxiv_id: "{arxiv_id}"')
    if doi:
        lines.append(f'doi: "{doi}"')
    if authors:
        lines.append("authors:")
        for a in authors:
            lines.append(f"  - {a}")
    lines.append("tags:")
    lines.append("  - paper")
    lines.append(f"citations: {citations}")
    lines.append("sources:")
    lines.append(f'  ocr: ".grounding/md_fc/{slug}.md"')
    lines.append("  drafted_by: deepseek")
    lines.append("  reviewed_by: grok")
    lines.append(f'ingest_date: "{today}"')
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def xai_integrate(deepseek_summary: str, metadata: dict, source_info: dict, slug: str) -> str:
    """Final integration step. Tries XAI Grok, then DeepSeek passthrough."""
    title = metadata.get("title") or slug
    abs_url = source_info.get("abs_url") or source_info.get("pdf_url")

    system = (
        "You are the editor for the Cardio Wiki, an interventional cardiology intelligence system. "
        "Take the DeepSeek summary below and produce a clean, Markdown-only wiki page body "
        "(NO YAML frontmatter — that will be prepended separately). "
        "Start with a level-1 heading using the paper title, then sections: "
        "Clinical Question, PICO (table), Key Results, What Changed?, What Did Not Change?, "
        "How I Would Explain This to a Patient, Relevance for Indian Practice, Related Trials. "
        "Be clinically precise; do not invent claims. Keep paragraphs tight."
    )
    user = (
        f"Title: {title}\nLink: {abs_url}\n\n"
        f"--- DeepSeek summary ---\n{deepseek_summary}"
    )

    xai_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if xai_key:
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {xai_key}", "Content-Type": "application/json"},
                json={
                    "model": os.environ.get("XAI_INGEST_MODEL", "grok-3"),
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 6000,
                    "temperature": 0.2,
                },
                timeout=300,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # Fallback: return DeepSeek output as the final body.
    return deepseek_summary


# ---------------------------------------------------------------------------
# 6. Pipeline driver (yields events)
# ---------------------------------------------------------------------------


def _event(name: str, **payload) -> Tuple[str, dict]:
    return (name, payload)


def ingest_pipeline(
    *,
    url: str | None = None,
    pdf_bytes: bytes | None = None,
    title_hint: str | None = None,
    slug_hint: str | None = None,
) -> Generator[Tuple[str, dict], None, None]:
    try:
        # ── 1. Detect / acquire ────────────────────────────────────────────
        if pdf_bytes:
            yield _event("status", stage="source", message=f"Using uploaded PDF ({len(pdf_bytes)} bytes)")
            source_info = {"kind": "upload", "id": None, "pdf_url": None, "abs_url": None}
        else:
            if not url:
                raise ValueError("Provide either a URL or upload a PDF")
            yield _event("status", stage="source", message=f"Detecting source from URL")
            source_info = detect_source(url)
            yield _event("source", **source_info)

        # ── 2. Mistral OCR ────────────────────────────────────────────────
        yield _event("status", stage="ocr", message="Mistral OCR — extracting paper text…")
        if pdf_bytes:
            ocr_md = mistral_ocr_bytes(pdf_bytes)
        else:
            try:
                ocr_md = mistral_ocr_url(source_info["pdf_url"])
            except Exception as e:
                yield _event("status", stage="ocr", message=f"Direct OCR failed ({type(e).__name__}); downloading and uploading…")
                pdf = requests.get(source_info["pdf_url"], headers={"User-Agent": "Mozilla/5.0", "Accept": "application/pdf"}, timeout=120).content
                if len(pdf) < 30000:
                    raise RuntimeError(f"PDF download too small ({len(pdf)} bytes) — likely blocked")
                ocr_md = mistral_ocr_bytes(pdf)
        yield _event("status", stage="ocr", message=f"OCR complete — {len(ocr_md):,} chars")

        ocr_title = title_hint
        if not ocr_title:
            for line in ocr_md.splitlines()[:20]:
                line = line.strip().lstrip("# ").strip()
                if 8 <= len(line) <= 200 and not line.lower().startswith(("abstract", "doi", "preprint", "received")):
                    ocr_title = line
                    break

        # ── 3. Citation lookup ────────────────────────────────────────────
        yield _event("status", stage="citation", message="Semantic Scholar — looking up metadata + citations…")
        metadata = None
        if source_info.get("kind") == "arxiv":
            metadata = semantic_scholar_lookup(arxiv_id=source_info["id"])
        if not metadata and source_info.get("kind") == "doi":
            metadata = semantic_scholar_lookup(doi=source_info["id"])
        if not metadata and ocr_title:
            metadata = semantic_scholar_lookup(title=ocr_title)
        if not metadata:
            metadata = {"title": ocr_title or "Untitled paper", "authors": [], "year": None,
                        "venue": None, "citations": 0, "abstract": "", "external_ids": {}}
            yield _event("status", stage="citation", message="Semantic Scholar miss — trying Grok web-search…")
            est = xai_citation_estimate(metadata["title"])
            if est is not None:
                metadata["citations"] = est
                yield _event("status", stage="citation", message=f"Grok estimate: ~{est} citations")
        else:
            yield _event("status", stage="citation",
                         message=f"Found: {metadata.get('title','?')[:80]} · {metadata.get('citations',0)} citations")
        yield _event("metadata", **{k: v for k, v in metadata.items() if k != "abstract"})

        # ── 4. DeepSeek summarization ─────────────────────────────────────
        yield _event("status", stage="summary", message="DeepSeek — producing structured summary (this takes 1-3 min)…")
        deepseek_summary = deepseek_summarize(ocr_md, metadata)
        yield _event("status", stage="summary", message=f"Summary done — {len(deepseek_summary):,} chars")

        # ── 5. Final integration via XAI Grok ────────────────────────────
        slug = slug_hint or _slugify(metadata.get("title") or ocr_title or "paper")
        backend = "xai-grok" if (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")) else "deepseek-passthrough"
        yield _event("status", stage="integrate", message=f"Final integration via {backend}…")
        body = xai_integrate(deepseek_summary, metadata, source_info, slug)
        yield _event("status", stage="integrate", message=f"Integration done — {len(body):,} chars")

        # ── 6. Write files ────────────────────────────────────────────────
        GROUNDING_DIR.mkdir(parents=True, exist_ok=True)
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        ocr_path = GROUNDING_DIR / f"{slug}.md"
        ocr_path.write_text(ocr_md, encoding="utf-8")

        page_path = PAPERS_DIR / f"{slug}.md"
        if page_path.exists():
            slug = f"{slug}-v{int(time.time()) % 10000}"
            page_path = PAPERS_DIR / f"{slug}.md"
            ocr_path = GROUNDING_DIR / f"{slug}.md"
            ocr_path.write_text(ocr_md, encoding="utf-8")

        full_page = _build_frontmatter(metadata, slug, source_info) + body.lstrip("\n")
        page_path.write_text(full_page, encoding="utf-8")

        yield _event("status", stage="write", message=f"Wrote {page_path.relative_to(BASE_DIR)}")
        yield _event("done", slug=slug, page_path=str(page_path.relative_to(BASE_DIR)),
                     ocr_path=str(ocr_path.relative_to(BASE_DIR)),
                     citations=metadata.get("citations", 0),
                     title=metadata.get("title") or slug)
    except Exception as exc:
        yield _event("error", message=f"{type(exc).__name__}: {exc}")
