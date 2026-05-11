#!/usr/bin/env python3
"""Batch ingest trials through the Mistral/DeepSeek/XAI pipeline.

Reads trial URL list from TRIAL_URLS below, runs the pipeline for each,
and prints success/failure to stdout. Failures are logged but don't
stop the batch.

Usage:
    python batch_ingest.py                  # run all
    python batch_ingest.py --limit 3        # first 3 only
    python batch_ingest.py --slug orbita-2  # one specific trial
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Load .env so MISTRAL_KEY etc. are available
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

from paper_ingest import ingest_pipeline


# (slug, url) — URLs prioritize PMC open-access > publisher full-text > PubMed page
TRIAL_URLS: list[tuple[str, str]] = [
    # ── Stable CAD ──
    ("ischemia",     "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7274550/"),
    ("courage",      "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2989250/"),
    ("orbita",       "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(17)32714-9/fulltext"),
    ("orbita-2",     "https://www.nejm.org/doi/full/10.1056/NEJMoa2310610"),
    ("syntax",       "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4380200/"),
    ("excel",        "https://www.nejm.org/doi/full/10.1056/NEJMoa1610227"),
    ("noble",        "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(16)32052-9/fulltext"),
    # ── ACS / STEMI ──
    ("complete",     "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7220899/"),
    ("flower-mi",    "https://www.nejm.org/doi/full/10.1056/NEJMoa2104650"),
    ("danami-3-primulti", "https://www.nejm.org/doi/full/10.1056/NEJMoa1507995"),
    # ── Imaging-Guided PCI ──
    ("ultimate",     "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6309398/"),
    ("ilumien-iv",   "https://www.nejm.org/doi/full/10.1056/NEJMoa2305861"),
    ("october",      "https://www.nejm.org/doi/full/10.1056/NEJMoa2307770"),
    # ── Physiology-Guided PCI ──
    ("fame",         "https://www.nejm.org/doi/full/10.1056/NEJMoa0807611"),
    ("fame-2",       "https://www.nejm.org/doi/full/10.1056/NEJMoa1205361"),
    ("define-flair", "https://www.nejm.org/doi/full/10.1056/NEJMoa1700445"),
    ("ifr-swedeheart","https://www.nejm.org/doi/full/10.1056/NEJMoa1616540"),
    # ── Antiplatelets ──
    ("twilight",     "https://www.nejm.org/doi/full/10.1056/NEJMoa1908419"),
    ("master-dapt",  "https://www.nejm.org/doi/full/10.1056/NEJMoa2108749"),
    ("stopdapt-2",   "https://jamanetwork.com/journals/jama/fullarticle/2737535"),
    ("global-leaders","https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(18)31858-0/fulltext"),
    ("host-exam",    "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)01445-8/fulltext"),
    # ── AF + PCI ──
    ("augustus",     "https://www.nejm.org/doi/full/10.1056/NEJMoa1817083"),
    ("pioneer-af-pci","https://www.nejm.org/doi/full/10.1056/NEJMoa1611594"),
    ("re-dual-pci",  "https://www.nejm.org/doi/full/10.1056/NEJMoa1708454"),
    ("entrust-af-pci","https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(19)31872-0/fulltext"),
    ("aristotle",    "https://www.nejm.org/doi/full/10.1056/NEJMoa1107039"),
    # ── TAVR / Structural ──
    ("partner-3",    "https://www.nejm.org/doi/full/10.1056/NEJMoa1814052"),
    ("early-tavr",   "https://www.nejm.org/doi/full/10.1056/NEJMoa2405880"),
    ("dedicate",     "https://www.nejm.org/doi/full/10.1056/NEJMoa2400685"),
    # ── Lipid / Prevention ──
    ("fourier",      "https://www.nejm.org/doi/full/10.1056/NEJMoa1615664"),
    ("odyssey-outcomes","https://www.nejm.org/doi/full/10.1056/NEJMoa1801174"),
    ("clear-outcomes","https://www.nejm.org/doi/full/10.1056/NEJMoa2215024"),
    ("select",       "https://www.nejm.org/doi/full/10.1056/NEJMoa2307563"),
    # ── Heart Failure ──
    ("dapa-hf",      "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6816069/"),
    ("emperor-reduced","https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7745179/"),
    ("emperor-preserved","https://www.nejm.org/doi/full/10.1056/NEJMoa2107038"),
    ("deliver",      "https://www.nejm.org/doi/full/10.1056/NEJMoa2206286"),
    ("strong-hf",    "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)02076-1/fulltext"),
]


def run_one(slug: str, url: str) -> dict:
    print(f"\n{'='*80}")
    print(f"[{slug}]  {url}")
    print(f"{'='*80}")

    result = {"slug": slug, "url": url, "success": False, "error": None,
              "page_path": None, "citations": 0}
    try:
        for event_name, payload in ingest_pipeline(url=url, slug_hint=slug):
            if event_name == "status":
                stage = payload.get("stage", "?")
                msg = payload.get("message", "")
                print(f"  [{stage}] {msg}")
            elif event_name == "metadata":
                title = payload.get("title", "?")[:80]
                cits = payload.get("citations", 0)
                print(f"  [metadata] {title} · {cits} citations")
                result["citations"] = cits
            elif event_name == "done":
                print(f"  [done] {payload.get('page_path')}")
                result["success"] = True
                result["page_path"] = payload.get("page_path")
                result["citations"] = payload.get("citations", 0)
            elif event_name == "error":
                print(f"  [ERROR] {payload.get('message')}")
                result["error"] = payload.get("message")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [PIPELINE ERROR] {result['error']}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Run only first N trials")
    parser.add_argument("--slug", help="Run a single trial by slug")
    parser.add_argument("--start", type=int, default=0, help="Start at index N")
    args = parser.parse_args()

    targets = TRIAL_URLS
    if args.slug:
        targets = [(s, u) for (s, u) in TRIAL_URLS if s == args.slug]
        if not targets:
            print(f"No trial with slug '{args.slug}'")
            sys.exit(1)
    elif args.limit:
        targets = TRIAL_URLS[args.start : args.start + args.limit]

    print(f"Running pipeline for {len(targets)} trials")

    results = []
    for slug, url in targets:
        results.append(run_one(slug, url))
        # gentle pause between calls to avoid rate limits
        time.sleep(2)

    # Summary
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
            print(f"  ✓ {r['slug']:20s}  {r['page_path']}  ({r['citations']} citations)")
    if fail:
        print("\nFailed (likely paywall / OCR block):")
        for r in fail:
            print(f"  ✗ {r['slug']:20s}  {r['error']}")


if __name__ == "__main__":
    main()
