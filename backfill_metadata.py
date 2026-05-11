#!/usr/bin/env python3
"""Backfill year and venue in already-ingested wiki pages from cached OCR + Grok.

Loads each wiki/sources/papers/*.md, reads its cached OCR from .grounding/md_fc/,
runs extract_metadata_from_ocr to recover year/venue, falls back to xai_year_estimate
if needed, and rewrites the frontmatter in place.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from paper_ingest import extract_metadata_from_ocr, xai_year_estimate

PAPERS = BASE_DIR / "wiki" / "sources" / "papers"
OCR_DIR = BASE_DIR / ".grounding" / "md_fc"

FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
FIELD_RE = lambda f: re.compile(rf"^({re.escape(f)}:\s*)(.*)$", re.MULTILINE)


def update_field(fm: str, field: str, new_value) -> str:
    pat = FIELD_RE(field)
    rendered = "null" if new_value is None else (
        f'"{new_value}"' if isinstance(new_value, str) else str(new_value)
    )
    if pat.search(fm):
        return pat.sub(lambda m: f"{m.group(1)}{rendered}", fm)
    return fm.rstrip() + f"\n{field}: {rendered}\n"


def get_field(fm: str, field: str) -> str | None:
    m = FIELD_RE(field).search(fm)
    if not m:
        return None
    val = m.group(2).strip().strip('"').strip("'")
    return val if val and val.lower() not in ("null", "none", "") else None


for md_file in sorted(PAPERS.glob("*.md")):
    text = md_file.read_text(encoding="utf-8")
    fm_match = FM_RE.match(text)
    if not fm_match:
        print(f"  skip (no frontmatter): {md_file.name}")
        continue
    fm = fm_match.group(1)
    body = text[fm_match.end():]

    title = get_field(fm, "title") or md_file.stem

    # Always re-extract from OCR cache (overwriting stale values)
    ocr_file = OCR_DIR / md_file.name
    extracted = {"year": None, "venue": None}
    if ocr_file.exists():
        ocr = ocr_file.read_text(encoding="utf-8")
        extracted = extract_metadata_from_ocr(ocr)

    new_year = extracted.get("year")
    new_venue = extracted.get("venue")

    if not new_year:
        print(f"  asking Grok for year of: {title[:60]}")
        new_year = xai_year_estimate(title)

    # Always overwrite — clear stale wrong values too
    fm = update_field(fm, "year", int(new_year) if new_year else None)
    fm = update_field(fm, "venue", new_venue)
    md_file.write_text(f"---\n{fm}\n---\n{body}", encoding="utf-8")
    print(f"  → {md_file.name}: year={new_year} venue={new_venue}")
