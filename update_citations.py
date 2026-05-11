#!/usr/bin/env python3
"""Update citation counts for all wiki papers via Semantic Scholar batch API.

Usage:
    python update_citations.py
    python update_citations.py --dry-run
    python update_citations.py --stale-days 7
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "wiki" / "sources" / "papers"
TRIALS_DIRS = [
    BASE_DIR / "wiki" / "trials" / sub
    for sub in [
        "stable-cad", "acs-stemi", "imaging-guided-pci",
        "physiology-guided-pci", "antiplatelets", "af-pci",
        "tavr-structural", "lipid-prevention", "heart-failure",
    ]
]
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
BATCH_SIZE = 500
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "citationCount,title,year,venue"


def extract_frontmatter(text: str) -> dict[str, Any]:
    import yaml
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}


def update_frontmatter_field(text: str, field: str, new_value: Any) -> str:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return text

    fm_text = match.group(1)
    pattern = re.compile(rf"^({re.escape(field)}:\s*)(.+)$", re.MULTILINE)
    field_match = pattern.search(fm_text)

    if field_match:
        new_fm = pattern.sub(rf"\g<1>{new_value}", fm_text)
    else:
        new_fm = fm_text.rstrip() + f"\n{field}: {new_value}"

    return text[: match.start(1)] + new_fm + text[match.end(1) :]


def collect_papers() -> list[dict[str, Any]]:
    papers = []
    search_dirs = [PAPERS_DIR] + TRIALS_DIRS
    for d in search_dirs:
        if not d.exists():
            continue
        for md_file in sorted(d.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            meta = extract_frontmatter(text)
            paper_id = meta.get("arxiv_id") or meta.get("doi") or meta.get("pmid")
            if not paper_id or paper_id == "null":
                continue
            paper_id = str(paper_id).strip().strip('"').strip("'")
            papers.append({
                "file": md_file,
                "paper_id": paper_id,
                "id_type": "arxiv" if meta.get("arxiv_id") else ("doi" if meta.get("doi") else "pmid"),
                "title": meta.get("title", md_file.stem),
                "current_citations": meta.get("citations", 0),
                "last_updated": meta.get("citations_updated"),
            })
    return papers


def batch_fetch_citations(papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results = {}
    for i in range(0, len(papers), BATCH_SIZE):
        batch = papers[i : i + BATCH_SIZE]
        s2_ids = []
        for p in batch:
            if p["id_type"] == "arxiv":
                s2_ids.append(f"ArXiv:{p['paper_id']}")
            elif p["id_type"] == "doi":
                s2_ids.append(f"DOI:{p['paper_id']}")
            else:
                s2_ids.append(f"PMID:{p['paper_id']}")

        try:
            resp = requests.post(
                S2_BATCH_URL,
                params={"fields": S2_FIELDS},
                json={"ids": s2_ids},
                timeout=30,
            )
            if resp.status_code == 429:
                print(f"  Rate limited, waiting 60s...")
                time.sleep(60)
                resp = requests.post(
                    S2_BATCH_URL,
                    params={"fields": S2_FIELDS},
                    json={"ids": s2_ids},
                    timeout=30,
                )

            if resp.status_code != 200:
                print(f"  S2 batch error HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            data = resp.json()
            for j, item in enumerate(data):
                if item is None:
                    continue
                pid = batch[j]["paper_id"]
                results[pid] = {
                    "citationCount": item.get("citationCount", 0),
                    "title": item.get("title"),
                    "year": item.get("year"),
                    "venue": item.get("venue"),
                }
        except requests.RequestException as e:
            print(f"  Request error: {e}")

        if i + BATCH_SIZE < len(papers):
            time.sleep(1.5)

    return results


def run_update(dry_run: bool = False, stale_days: int | None = None) -> dict[str, Any]:
    papers = collect_papers()
    print(f"Found {len(papers)} papers with identifiers")

    if stale_days is not None:
        cutoff = (datetime.now() - timedelta(days=stale_days)).isoformat()[:10]
        papers = [
            p for p in papers
            if not p["last_updated"] or str(p["last_updated"]) < cutoff
        ]
        print(f"  {len(papers)} papers stale (>{stale_days} days since last update)")

    if not papers:
        print("Nothing to update.")
        return {"total": 0, "updated": 0, "unchanged": 0, "errors": 0}

    print(f"Fetching citations for {len(papers)} papers from Semantic Scholar...")
    s2_data = batch_fetch_citations(papers)
    print(f"  Got data for {len(s2_data)}/{len(papers)} papers")

    updated = 0
    unchanged = 0
    not_found = 0
    changes = []
    today = datetime.now().strftime("%Y-%m-%d")

    for paper in papers:
        pid = paper["paper_id"]
        s2 = s2_data.get(pid)
        if not s2:
            not_found += 1
            continue

        new_count = s2["citationCount"]
        old_count = paper["current_citations"]
        try:
            old_count = int(old_count)
        except (ValueError, TypeError):
            old_count = 0

        if new_count != old_count:
            delta = new_count - old_count
            changes.append({
                "title": paper["title"][:60],
                "paper_id": pid,
                "old": old_count,
                "new": new_count,
                "delta": delta,
            })

            if not dry_run:
                text = paper["file"].read_text(encoding="utf-8")
                text = update_frontmatter_field(text, "citations", new_count)
                text = update_frontmatter_field(text, "citations_updated", f'"{today}"')
                paper["file"].write_text(text, encoding="utf-8")

            updated += 1
        else:
            if not dry_run:
                text = paper["file"].read_text(encoding="utf-8")
                text = update_frontmatter_field(text, "citations_updated", f'"{today}"')
                paper["file"].write_text(text, encoding="utf-8")
            unchanged += 1

    print(f"\n{'='*80}")
    print(f"  Citation Update {'(DRY RUN) ' if dry_run else ''}Complete")
    print(f"{'='*80}")
    print(f"  Total papers:  {len(papers)}")
    print(f"  Updated:       {updated}")
    print(f"  Unchanged:     {unchanged}")
    print(f"  Not in S2:     {not_found}")

    if changes:
        print(f"\n{'Delta':>7}  {'Old':>7}  {'New':>7}  Paper")
        print(f"{'-'*80}")
        for c in sorted(changes, key=lambda x: -abs(x["delta"])):
            sign = "+" if c["delta"] > 0 else ""
            print(f"{sign}{c['delta']:>6}  {c['old']:>7}  {c['new']:>7}  {c['title']}")

    return {
        "total": len(papers),
        "updated": updated,
        "unchanged": unchanged,
        "not_found": not_found,
        "changes": changes,
    }


def register_citation_routes(app):
    from flask import jsonify, session as flask_session

    @app.route("/api/update-citations", methods=["POST"])
    def api_update_citations():
        if not flask_session.get("authed"):
            return jsonify({"error": "unauthorized"}), 401
        result = run_update(dry_run=False, stale_days=1)
        return jsonify(result)

    @app.route("/api/update-citations/dry-run", methods=["GET"])
    def api_update_citations_dry():
        if not flask_session.get("authed"):
            return jsonify({"error": "unauthorized"}), 401
        result = run_update(dry_run=True)
        return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update citation counts for wiki papers.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
    parser.add_argument("--stale-days", type=int, help="Only update papers not updated in N days.")
    args = parser.parse_args()
    run_update(dry_run=args.dry_run, stale_days=args.stale_days)
