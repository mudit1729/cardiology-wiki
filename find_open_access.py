#!/usr/bin/env python3
"""For each cardiology trial, query PubMed → check PMC open-access → output verified PDF URLs.

Uses NCBI E-utilities (no API key required). Conservative: only outputs trials
where we can confirm the PMC paper is in the OA subset (real PDF download available).

Output: trial_urls.json with {slug: {pmid, doi, pmcid, pdf_url, title, year}}
        Trials without OA access are saved with pdf_url=null so we know to skip them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

# Trial → PubMed search query (trial acronym + first author + year)
# Queries chosen to be specific enough to disambiguate
TRIAL_QUERIES = {
    # Stable CAD
    "ischemia":              '"Initial Invasive or Conservative Strategy for Stable Coronary Disease" Maron 2020',
    "courage":               '"Optimal medical therapy with or without PCI for stable coronary disease" Boden 2007',
    "orbita":                '"Percutaneous coronary intervention in stable angina" Al-Lamee 2018',
    "orbita-2":              '"placebo-controlled trial of percutaneous coronary intervention" Rajkumar 2023',
    "syntax":                '"Percutaneous coronary intervention versus coronary-artery bypass" Serruys 2009',
    "excel":                 '"Everolimus-Eluting Stents or Bypass Surgery for Left Main Coronary Disease" Stone',
    "noble":                 '"Percutaneous coronary angioplasty versus coronary artery bypass grafting in treatment of unprotected left main stenosis" Makikallio 2016',
    # ACS / STEMI
    "complete":              '"Complete Revascularization with Multivessel PCI for Myocardial Infarction" Mehta 2019',
    "flower-mi":             '"Flow Evaluation to Guide Revascularization in Multivessel ST-Elevation" Puymirat 2021',
    "danami-3-primulti":     '"Complete revascularisation versus treatment of the culprit lesion only" Engstrom 2015',
    # Imaging-Guided PCI
    "ultimate":              '"Intravascular Ultrasound-Guided Drug-Eluting Stent Implantation" Zhang ULTIMATE 2018',
    "ilumien-iv":            '"Optical Coherence Tomography-Guided Coronary Stent" Ali 2023',
    "october":               '"OCT or Angiography Guidance for PCI in Complex Bifurcation Lesions" Holm 2023',
    # Physiology-Guided PCI
    "fame":                  '"Fractional flow reserve versus angiography for guiding percutaneous coronary intervention" Tonino 2009',
    "fame-2":                '"Fractional flow reserve-guided PCI versus medical therapy" De Bruyne 2012',
    "define-flair":          '"Use of the Instantaneous Wave-free Ratio or Fractional Flow Reserve" Davies 2017',
    "ifr-swedeheart":        '"Instantaneous Wave-free Ratio versus Fractional Flow Reserve" Gotberg 2017',
    # Antiplatelets
    "twilight":              '"Ticagrelor with or without Aspirin in High-Risk Patients" Mehran 2019',
    "master-dapt":           '"Dual Antiplatelet Therapy after PCI in Patients at High Bleeding Risk" Valgimigli 2021',
    "stopdapt-2":            '"Effect of 1-Month Dual Antiplatelet Therapy Followed by Clopidogrel" Watanabe 2019',
    "global-leaders":        '"Ticagrelor plus aspirin for 1 month, followed by ticagrelor monotherapy" Vranckx 2018',
    "host-exam":             '"Clopidogrel versus aspirin monotherapy" Koo HOST-EXAM 2021',
    # AF + PCI
    "augustus":              '"Antithrombotic Therapy after Acute Coronary Syndrome or PCI in Atrial Fibrillation" Lopes 2019',
    "pioneer-af-pci":        '"Prevention of Bleeding in Patients with Atrial Fibrillation Undergoing PCI" Gibson 2016',
    "re-dual-pci":           '"Dual Antithrombotic Therapy with Dabigatran after PCI" Cannon 2017',
    "entrust-af-pci":        '"Edoxaban-based versus vitamin K antagonist-based antithrombotic regimen" Vranckx 2019',
    "aristotle":             '"Apixaban versus warfarin in patients with atrial fibrillation" Granger 2011',
    # TAVR / Structural
    "partner-3":             '"Transcatheter Aortic-Valve Replacement with a Balloon-Expandable Valve in Low-Risk" Mack 2019',
    "early-tavr":            '"Transcatheter Aortic-Valve Replacement in Asymptomatic Severe Aortic Stenosis" Genereux 2024',
    "dedicate":              '"Transcatheter or Surgical Aortic-Valve Replacement in Lower-Risk" Blankenberg 2024',
    # Lipid / Prevention
    "fourier":               '"Evolocumab and Clinical Outcomes in Patients with Cardiovascular Disease" Sabatine 2017',
    "odyssey-outcomes":      '"Alirocumab and Cardiovascular Outcomes after Acute Coronary Syndrome" Schwartz 2018',
    "clear-outcomes":        '"Bempedoic Acid and Cardiovascular Outcomes in Statin-Intolerant Patients" Nissen 2023',
    "select":                '"Semaglutide and Cardiovascular Outcomes in Obesity without Diabetes" Lincoff 2023',
    # Heart Failure
    "dapa-hf":               '"Dapagliflozin in Patients with Heart Failure and Reduced Ejection Fraction" McMurray 2019',
    "emperor-reduced":       '"Cardiovascular and Renal Outcomes with Empagliflozin in Heart Failure" Packer 2020',
    "emperor-preserved":     '"Empagliflozin in Heart Failure with a Preserved Ejection Fraction" Anker 2021',
    "deliver":               '"Dapagliflozin in Heart Failure with Mildly Reduced or Preserved Ejection" Solomon 2022',
    "strong-hf":             '"Safety, tolerability, and efficacy of up-titration of guideline-directed medical therapies" Mebazaa 2022',
}

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


def esearch_pmid(query: str) -> str | None:
    """Return the first PMID matching this query, or None."""
    r = requests.get(f"{EUTILS}/esearch.fcgi", params={
        "db": "pubmed", "term": query, "retmax": 1, "retmode": "json",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


def efetch_summary(pmid: str) -> dict:
    """Return {title, year, doi} for a PMID. Uses XML to avoid JSON control-char issues."""
    import re as _re
    r = requests.get(f"{EUTILS}/esummary.fcgi", params={
        "db": "pubmed", "id": pmid, "version": "2.0",
    }, timeout=30)
    r.raise_for_status()
    xml = r.text
    title_m = _re.search(r"<Name>Title</Name>.*?<Item[^>]*>([^<]+)</Item>", xml, _re.S)
    if not title_m:
        title_m = _re.search(r"<Title>([^<]+)</Title>", xml)
    title = title_m.group(1) if title_m else ""
    pubdate_m = _re.search(r"<Name>PubDate</Name>.*?<Item[^>]*>([^<]+)</Item>", xml, _re.S)
    if not pubdate_m:
        pubdate_m = _re.search(r"<PubDate>([^<]+)</PubDate>", xml)
    year = (pubdate_m.group(1) if pubdate_m else "")[:4]
    doi = ""
    doi_m = _re.search(r"<ELocationID[^>]*EIdType=\"doi\"[^>]*>([^<]+)</ELocationID>", xml)
    if doi_m:
        doi = doi_m.group(1)
    else:
        doi_m = _re.search(r"<ArticleId IdType=\"doi\">([^<]+)</ArticleId>", xml)
        if doi_m:
            doi = doi_m.group(1)
    return {"title": title.strip(), "year": year, "doi": doi}


def elink_to_pmc(pmid: str) -> str | None:
    """Return PMC ID (e.g. 'PMC7220899') if the PubMed paper has a PMC version."""
    r = requests.get(f"{EUTILS}/elink.fcgi", params={
        "dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    linksets = data.get("linksets", [])
    if not linksets:
        return None
    for link in linksets[0].get("linksetdbs", []):
        if link.get("dbto") == "pmc":
            ids = link.get("links", [])
            if ids:
                return f"PMC{ids[0]}"
    return None


def oa_pdf_url(pmcid: str) -> str | None:
    """Return the open-access PDF URL for a PMC paper, or None if not in OA subset."""
    # Strip 'PMC' prefix if present (oa.fcgi accepts either form)
    pmc_num = pmcid.replace("PMC", "")
    r = requests.get(OA_API, params={"id": f"PMC{pmc_num}"}, timeout=30)
    if r.status_code != 200:
        return None
    text = r.text
    # OA API returns XML; look for href in the link element
    if "idDoesNotExist" in text or "is not Open Access" in text:
        return None
    # Find <link href="..." format="pdf" ...>
    import re
    m = re.search(r'<link[^>]*format="pdf"[^>]*href="([^"]+)"', text)
    if not m:
        # Try alternative: href before format
        m = re.search(r'href="([^"]+)"[^>]*format="pdf"', text)
    if m:
        url = m.group(1)
        # OA returns ftp:// — switch to https
        url = url.replace("ftp://", "https://")
        return url
    return None


def resolve_trial(slug: str, query: str) -> dict:
    print(f"  [{slug}] searching: {query}")
    out = {"slug": slug, "query": query, "pmid": None, "title": None,
           "year": None, "doi": None, "pmcid": None, "pdf_url": None,
           "status": "pending"}
    try:
        pmid = esearch_pmid(query)
        if not pmid:
            out["status"] = "no_pubmed_match"
            print(f"    → no PubMed match")
            return out
        out["pmid"] = pmid
        time.sleep(0.4)  # NCBI rate limit ~3 req/sec without API key

        meta = efetch_summary(pmid)
        out.update(meta)
        print(f"    PMID {pmid} | {meta['title'][:70]}")
        time.sleep(0.4)

        pmcid = elink_to_pmc(pmid)
        if not pmcid:
            out["status"] = "not_in_pmc"
            print(f"    → not in PMC (paywalled)")
            return out
        out["pmcid"] = pmcid
        time.sleep(0.4)

        pdf_url = oa_pdf_url(pmcid)
        if not pdf_url:
            out["status"] = "in_pmc_not_oa"
            print(f"    → {pmcid} is in PMC but not OA subset")
            return out
        out["pdf_url"] = pdf_url
        out["status"] = "ok"
        print(f"    ✓ OA PDF: {pdf_url}")
    except Exception as e:
        out["status"] = f"error:{type(e).__name__}"
        print(f"    ERROR: {e}")
    return out


def main():
    results = {}
    for slug, query in TRIAL_QUERIES.items():
        results[slug] = resolve_trial(slug, query)
        time.sleep(0.5)

    out_path = Path(__file__).parent / "trial_urls.json"
    out_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    by_status = {}
    for slug, r in results.items():
        by_status.setdefault(r["status"], []).append(slug)
    for status, slugs in sorted(by_status.items()):
        print(f"\n{status}: {len(slugs)}")
        for s in slugs:
            print(f"  - {s}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
