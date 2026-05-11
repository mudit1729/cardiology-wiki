#!/usr/bin/env python3
"""Batch-ingest verified OA cardiology RCTs from PMC.

For each (slug, pmcid) pair, downloads the PDF via Europe PMC's render endpoint
(which works when NCBI's OA FTP URLs return 404), then feeds the bytes to
ingest_pipeline. Skips and logs any paper that fails OCR or returns non-PDF bytes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent

# Load .env
env_path = BASE_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from paper_ingest import ingest_pipeline


# (slug, pmcid, title_hint) — verified to have direct PDFs via Europe PMC
PAPERS: list[tuple[str, str, str]] = [
    ("oct-pci-decision-making-2015", "PMC4677272",
     "Optical coherence tomography imaging during PCI impacts physician decision-making"),
    ("ivus-pci-multicenter-2022", "PMC8740965",
     "Multi-center randomized trial of IVUS-guided PCI superiority"),
    ("colchicine-post-pci-2022", "PMC8739658",
     "Colchicine for symptomatic coronary artery disease after PCI"),
    ("antiplatelet-vs-anticoagulant-tavr-2023", "PMC10668142",
     "Single antiplatelet or anticoagulant followed by antiplatelet after TAVR"),
    ("biatrial-vs-left-atrial-ablation-rheumatic-mv-2022", "PMC9710358",
     "Bi-atrial versus left atrial ablation in rheumatic mitral valve disease and non-paroxysmal AF"),
]


def fetch_pdf(pmcid: str) -> bytes | None:
    """Download the PDF for a PMC paper via Europe PMC's render endpoint."""
    url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                         timeout=120, allow_redirects=True)
    except requests.RequestException as e:
        print(f"    download error: {e}")
        return None
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}")
        return None
    if not r.content[:4] == b"%PDF":
        print(f"    not a PDF (got {r.content[:100]!r})")
        return None
    return r.content


def run_one(slug: str, pmcid: str, title_hint: str) -> dict:
    print(f"\n{'='*80}")
    print(f"[{slug}]  {pmcid}")
    print(f"{'='*80}")

    out = {"slug": slug, "pmcid": pmcid, "success": False, "error": None,
           "page_path": None, "title": None, "citations": 0}

    print(f"  Fetching PDF from EuropePMC…")
    pdf = fetch_pdf(pmcid)
    if not pdf:
        out["error"] = "PDF fetch failed"
        return out
    print(f"  Got {len(pdf):,} bytes")

    try:
        for event_name, payload in ingest_pipeline(
            pdf_bytes=pdf, slug_hint=slug, title_hint=title_hint,
        ):
            if event_name == "status":
                stage = payload.get("stage", "?")
                msg = payload.get("message", "")
                print(f"  [{stage}] {msg}")
            elif event_name == "metadata":
                t = payload.get("title", "?")[:80]
                c = payload.get("citations", 0)
                print(f"  [metadata] {t} · {c} citations")
                out["citations"] = c
            elif event_name == "done":
                print(f"  [done] {payload.get('page_path')}")
                out["success"] = True
                out["page_path"] = payload.get("page_path")
                out["title"] = payload.get("title")
                out["citations"] = payload.get("citations", 0)
            elif event_name == "error":
                print(f"  [ERROR] {payload.get('message')}")
                out["error"] = payload.get("message")
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [PIPELINE ERROR] {out['error']}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Only first N")
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    targets = PAPERS[args.start:]
    if args.limit:
        targets = targets[:args.limit]

    print(f"Ingesting {len(targets)} papers")

    results = []
    for slug, pmcid, hint in targets:
        results.append(run_one(slug, pmcid, hint))
        time.sleep(2)

    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    succ = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    print(f"Succeeded: {len(succ)}/{len(results)}")
    print(f"Failed:    {len(fail)}/{len(results)}")
    if succ:
        print("\nWritten:")
        for r in succ:
            print(f"  ✓ {r['slug']:50s}  {r['page_path']}")
    if fail:
        print("\nFailed:")
        for r in fail:
            print(f"  ✗ {r['slug']:50s}  {r['error']}")


if __name__ == "__main__":
    main()
