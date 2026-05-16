"""Paper ingestion pipeline for the Cardio Wiki.

Stream events as we move through:
  1. Detect / acquire PDF (URL or upload)
  2. Mistral OCR  → ground-truth markdown
  3. Semantic Scholar lookup (with XAI fallback) → metadata + citations
  4. OpenAI GPT-5.5 streaming summarization → structured wiki draft
  5. GPT-5.5 final integration → polished wiki page
  6. Write wiki/sources/papers/{slug}.md + .grounding/md_fc/{slug}.md
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Generator, Tuple

import requests


_GIT_LOCK = threading.Lock()

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
    timeout = int(os.environ.get("XAI_CITATION_TIMEOUT", "8"))
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
                "model": "grok-4.2",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
                "temperature": 0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\d{1,7}", text)
        return int(m.group(0)) if m else None
    except Exception:
        return None


# Citation patterns in PDF headers (most reliable signal of publication year + venue).
# Matches "Open Heart 2022;9:e001887", "European Heart Journal (2015) 36, 3346-3355",
# "BMJ Open 2023;13:e076781", "Eur Heart J. 2015 Dec 14; 36(47):3346-3355", etc.
DOI_IN_OCR_RE = re.compile(r"\b(?:doi|DOI)[:\s]+\s*(10\.\d{4,9}/[^\s\)\]\>]+)", re.IGNORECASE)
DOI_BARE_RE = re.compile(r"\b(10\.\d{4,9}/[A-Za-z0-9._\-/]+[A-Za-z0-9])")
PMCID_IN_OCR_RE = re.compile(r"\b(PMC\d{5,10})\b")
OCR_TITLE_NOISE = {
    "open access",
    "protocol",
    "original research",
    "check for updates",
    "bmj",
}
OCR_TITLE_PREFIXES = (
    "abstract",
    "accepted",
    "author affiliations",
    "background",
    "correspondence",
    "doi",
    "funding",
    "preprint",
    "prepublication",
    "received",
    "to cite",
)
OCR_TITLE_JOURNAL_PREFIXES = (
    "bmj open ",
    "open heart ",
    "european heart journal ",
    "journal of the american college of cardiology ",
    "jacc ",
    "circulation ",
)


def extract_doi_from_ocr(ocr_md: str) -> str | None:
    head = ocr_md[:8000]
    m = DOI_IN_OCR_RE.search(head)
    if m:
        return m.group(1).rstrip(".,;").strip()
    m = DOI_BARE_RE.search(head)
    if m:
        return m.group(1).rstrip(".,;").strip()
    return None


def extract_pmcid_from_ocr(ocr_md: str) -> str | None:
    head = ocr_md[:8000]
    m = PMCID_IN_OCR_RE.search(head)
    return m.group(1) if m else None


def _clean_ocr_title_candidate(line: str) -> str:
    candidate = re.sub(r"\s+", " ", line.strip().lstrip("# ").strip())
    for prefix in OCR_TITLE_JOURNAL_PREFIXES:
        if candidate.lower().startswith(prefix) and len(candidate) > len(prefix) + 25:
            return candidate[len(prefix):].strip()
    return candidate


def extract_title_from_ocr(ocr_md: str) -> str | None:
    """Find the actual paper title near the OCR header, skipping journal boilerplate."""
    best: tuple[int, str] | None = None
    for raw in ocr_md.splitlines()[:80]:
        stripped = raw.strip()
        candidate = _clean_ocr_title_candidate(stripped)
        low = candidate.lower().strip(" :")
        if not candidate or low in OCR_TITLE_NOISE or low.startswith(OCR_TITLE_PREFIXES):
            continue
        if "doi:" in low or " et al." in low or "©" in candidate:
            continue
        word_count = len(re.findall(r"[A-Za-z][A-Za-z-]+", candidate))
        if word_count < 5 or not (25 <= len(candidate) <= 260):
            continue
        score = word_count
        if stripped.startswith("#"):
            score += 50
        if re.search(r"\b(randomi[sz]ed|trial|study|protocol|patients?)\b", low):
            score += 20
        if ":" in candidate:
            score += 5
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best else None


JOURNAL_CITE_PATTERNS = [
    # "Journal YEAR;Vol:..."  e.g. "Open Heart 2022;9:e001887"
    re.compile(r"\b([A-Z][A-Za-z]+(?:\s[A-Z]?[A-Za-z]+){0,5})\s+(\d{4})\s*[;:]\s*\d"),
    # "Journal (YEAR) Vol, ..."  e.g. "European Heart Journal (2015) 36, 3346"
    re.compile(r"\b([A-Z][A-Za-z]+(?:\s[A-Z]?[A-Za-z]+){0,5})\s+\((\d{4})\)\s+\d"),
    # "Journal Vol (YEAR) Pages"  e.g. "Indian Heart Journal 74 (2022) 363–368"
    re.compile(r"\b([A-Z][A-Za-z]+(?:\s[A-Z]?[A-Za-z]+){1,5})\s+\d+\s+\((\d{4})\)\s+\d"),
    # "Journal YEAR Mon DD;Vol(Iss):..." e.g. "Indian Heart J. 2022 Aug 22 Sep-Oct;"
    re.compile(r"\b([A-Z][A-Za-z]+(?:\s[A-Z]?[A-Za-z]+){0,5})\.\s+(\d{4})\s+[A-Z][a-z]{2,9}\s+\d"),
    # Citation in body: "Journal Vol(Iss): ePage" e.g. "PLoS ONE 17(1): e0260770"
    re.compile(r"\b([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+){0,5})\s+\d+\s*\(\s*\d+\s*\)\s*:\s*e?\d+"),
]
# Pattern to extract just the year when matched a venue-only pattern (PLOS ONE case)
PUBLISHED_DATE_RE = re.compile(r"Published:?\s+[A-Za-z]+\s+\d{1,2}[,\s]+(\d{4})", re.IGNORECASE)
# Publication year (prefer pub date, NOT received/accepted)
PUB_YEAR_RE = re.compile(
    r"(?:Published(?:\s+online)?|©|\(c\)|Copyright)[^\d]{0,50}(20[0-2]\d)",
    re.IGNORECASE,
)
# Year embedded in DOI suffix: "10.1136/openhrt-2021-001887" → 2021
DOI_YEAR_RE = re.compile(r"doi[:.\s]*10\.\d{4,9}/[a-zA-Z._\-]+[-_](20[0-2]\d)[-_]")
ANY_YEAR_RE = re.compile(r"\b(20[0-2]\d)\b")

# Filter out false-positive "venues" that are actually month names or other words
VENUE_BLACKLIST = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "received", "accepted", "published", "online", "abstract", "introduction",
    "methods", "results", "discussion", "conclusions", "background",
}


def _looks_like_venue(s: str) -> bool:
    s_low = s.strip().lower()
    if s_low in VENUE_BLACKLIST:
        return False
    if any(w in VENUE_BLACKLIST for w in s_low.split()):
        # First word is a month/section name - bad
        if s_low.split()[0] in VENUE_BLACKLIST:
            return False
    return True


def extract_metadata_from_ocr(ocr_md: str) -> dict:
    """Pull year and venue from the OCR header. Returns {year: int|None, venue: str|None}."""
    out: dict = {"year": None, "venue": None}
    head = ocr_md[:6000]

    # Try journal-citation patterns (highest signal for both year + venue)
    for pat in JOURNAL_CITE_PATTERNS:
        for m in pat.finditer(head):
            venue = m.group(1).rstrip(".").strip()
            if not _looks_like_venue(venue):
                continue
            year = None
            if m.lastindex and m.lastindex >= 2:
                try:
                    y = int(m.group(2))
                    if 1990 <= y <= 2030:
                        year = y
                except (ValueError, IndexError):
                    pass
            # Year-only fallback for venue-only patterns (e.g. "PLoS ONE 17(1): e0260770")
            if year is None:
                m2 = PUBLISHED_DATE_RE.search(head)
                if m2:
                    try:
                        y = int(m2.group(1))
                        if 1990 <= y <= 2030:
                            year = y
                    except ValueError:
                        pass
            if venue and year:
                out["venue"] = venue
                out["year"] = year
                return out
            elif venue and not out["venue"]:
                out["venue"] = venue  # keep searching for year

    # Year fallback: prefer "Published YEAR" / "© YEAR" over "Received YEAR"
    m = PUB_YEAR_RE.search(head)
    if m:
        try:
            y = int(m.group(1))
            if 1990 <= y <= 2030:
                out["year"] = y
                return out
        except ValueError:
            pass

    # Year fallback: extract from DOI suffix (e.g. "openhrt-2021-001887")
    m = DOI_YEAR_RE.search(head)
    if m:
        try:
            y = int(m.group(1))
            if 1990 <= y <= 2030:
                out["year"] = y
                return out
        except ValueError:
            pass

    # Last resort: most common year in the first 2000 chars
    years = ANY_YEAR_RE.findall(head[:2000])
    if years:
        from collections import Counter
        out["year"] = int(Counter(years).most_common(1)[0][0])
    return out


# Keyword → tag mapping for auto-tagging cardiology papers based on title + body content.
# Order matters only for the few keys where multiple keywords could match — most are independent.
TAG_KEYWORDS: list[tuple[str, list[str]]] = [
    ("stable-cad",            ["stable angina", "stable coronary", "chronic coronary"]),
    ("acs",                   ["acute coronary syndrome", "ACS"]),
    ("stemi",                 ["STEMI", "ST-elevation myocardial", "ST elevation"]),
    ("nstemi",                ["NSTEMI", "non-ST-elevation"]),
    ("pci",                   ["percutaneous coronary intervention", "PCI"]),
    ("imaging-guided-pci",    ["IVUS", "OCT", "intravascular ultrasound", "optical coherence"]),
    ("ivus",                  ["IVUS", "intravascular ultrasound"]),
    ("oct",                   ["OCT-guided", "OCT imaging", "optical coherence tomography"]),
    ("physiology-guided-pci", ["FFR", "iFR", "fractional flow reserve", "instantaneous wave-free"]),
    ("ffr",                   ["FFR", "fractional flow reserve"]),
    ("ifr",                   ["iFR", "instantaneous wave-free"]),
    ("antiplatelet",          ["antiplatelet", "DAPT", "aspirin", "ticagrelor", "clopidogrel", "prasugrel"]),
    ("dapt",                  ["DAPT", "dual antiplatelet"]),
    ("af-pci",                ["atrial fibrillation pci", "af pci", "undergoing pci", "percutaneous coronary intervention"]),
    ("atrial-fibrillation",   ["atrial fibrillation"]),
    ("anticoagulation",       ["anticoagulation", "anticoagulant", "warfarin", "DOAC"]),
    ("tavr",                  ["TAVR", "TAVI", "transcatheter aortic valve"]),
    ("structural-heart",      ["TAVR", "TAVI", "MitraClip", "transcatheter aortic", "transcatheter mitral", "valve replacement"]),
    ("aortic-stenosis",       ["aortic stenosis"]),
    ("mitral",                ["mitral valve", "MitraClip"]),
    ("rheumatic",             ["rheumatic"]),
    ("lipid-prevention",      ["PCSK9", "evolocumab", "alirocumab", "bempedoic", "ezetimibe", "icosapent", "LDL"]),
    ("pcsk9",                 ["PCSK9", "evolocumab", "alirocumab"]),
    ("statin",                ["statin"]),
    ("heart-failure",         ["heart failure", "HFrEF", "HFpEF", "ejection fraction"]),
    ("sglt2",                 ["SGLT2", "dapagliflozin", "empagliflozin", "canagliflozin"]),
    ("colchicine",            ["colchicine"]),
    ("ci-aki",                ["contrast-induced acute kidney", "CI-AKI", "contrast nephropathy"]),
    ("india-practice",        ["Indian", "India", "ABVIMS", "AIIMS", "south Asian"]),
    ("rct",                   ["randomized", "randomised"]),
    ("meta-analysis",         ["meta-analysis", "systematic review"]),
    ("rheumatic-heart-disease", ["rheumatic heart", "rheumatic mitral"]),
]

DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("guidelines-appropriate-use", ["guideline", "appropriate use", "appropriateness criteria", "consensus statement"]),
    ("india-lmic", ["india", "indian", "lmic", "low-income", "middle-income", "resource-limited"]),
    ("complications-safety", ["stent thrombosis", "restenosis", "bleeding", "acute kidney injury", "contrast-induced", "perforation", "no-reflow", "vascular complication", "safety alert"]),
    ("device-technology", ["drug-eluting stent", "drug-coated balloon", "scaffold", "atherectomy", "intravascular lithotripsy", "embolic protection", "closure device", "device"]),
    ("peripheral-endovascular", ["peripheral artery", "peripheral vascular", "carotid", "renal artery", "limb ischemia", "aortic", "endovascular"]),
    ("heart-failure-shock", ["cardiogenic shock", "mechanical circulatory support", "impella", "iabp", "ecmo", "heart failure", "high-risk pci"]),
    ("valve-rheumatic", ["rheumatic", "mitral valve", "mitral stenosis", "pbmv", "ptmc", "valve surgery"]),
    ("acs", ["acute coronary syndrome", "stemi", "nstemi", "primary pci", "post-mi"]),
    ("imaging-guided-pci", ["ivus", "oct", "intravascular ultrasound", "optical coherence", "imaging-guided"]),
    ("physiology-guided-pci", ["ffr", "ifr", "rfr", "cfr", "imr", "coronary physiology"]),
    ("antithrombotic", ["antiplatelet", "dapt", "p2y12", "aspirin", "ticagrelor", "clopidogrel", "prasugrel", "anticoagulation", "warfarin", "doac"]),
    ("structural-heart", ["tavr", "tavi", "teer", "mitraclip", "laao", "asd closure", "pfo closure", "paravalvular leak", "structural heart"]),
    ("prevention-lipids", ["statin", "ezetimibe", "pcsk9", "ldl", "lipoprotein", "lpa", "colchicine", "secondary prevention"]),
    ("ep-af", ["atrial fibrillation", "left atrial appendage", "ablation", "rhythm control"]),
    ("coronary-pci", ["pci", "percutaneous coronary intervention", "coronary", "left main", "bifurcation", "cto", "calcified lesion", "stable angina", "stable cad"]),
]


def auto_tag(title: str, body_excerpt: str = "", max_tags: int = 8) -> list[str]:
    """Return a list of taxonomy tags. Conservative: tag fires only when
    a keyword appears in the title OR ≥2 times in the body, AND is not
    in an exclusion-criteria context (e.g. "STEMI was excluded").
    """
    title_low = title.lower()
    body_low = body_excerpt[:12000].lower()
    matched: list[str] = ["paper"]
    combined_low = f"{title_low}\n{body_low}"

    def is_excluded(kw_low: str) -> bool:
        # If every body occurrence is in an exclusion-criteria context, skip the tag.
        positives = 0
        for m in re.finditer(rf"\b{re.escape(kw_low)}\b", body_low):
            window = body_low[max(0, m.start() - 80):m.start()]
            if re.search(r"exclud|not eligible|exception of", window):
                continue
            positives += 1
        return positives == 0

    for tag, keywords in TAG_KEYWORDS:
        if tag in matched:
            continue
        if tag == "af-pci" and not (
            re.search(r"\b(atrial fibrillation|af)\b", combined_low)
            and re.search(r"\b(pci|percutaneous coronary intervention)\b", combined_low)
        ):
            continue
        for kw in keywords:
            kw_low = kw.lower()
            in_title = re.search(rf"\b{re.escape(kw_low)}\b", title_low) is not None
            body_n = len(re.findall(rf"\b{re.escape(kw_low)}\b", body_low))
            if (in_title or body_n >= 2) and (in_title or not is_excluded(kw_low)):
                matched.append(tag)
                break
        if len(matched) >= max_tags:
            break
    return matched


def infer_domain(title: str, body: str, group_hint: str | None = None) -> str:
    if group_hint and group_hint in {key for key, _ in DOMAIN_KEYWORDS}:
        return group_hint
    text = f"{title}\n{body[:12000]}".lower()
    for domain, keywords in DOMAIN_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return domain
    return "other"


def xai_year_estimate(title: str) -> int | None:
    """Ask Grok for the publication year of a paper. Returns 4-digit int or None."""
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        return None
    timeout = int(os.environ.get("XAI_CITATION_TIMEOUT", "8"))
    prompt = (
        f"What year was the paper titled \"{title}\" published? "
        "Return ONLY a 4-digit year (1990-2030) or the word UNKNOWN. No other text."
    )
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "grok-4.2",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"\b(19[89]\d|20[0-3]\d)\b", text)
        return int(m.group(1)) if m else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 4. GPT-5.5 streaming summarization
# ---------------------------------------------------------------------------


def _strip_frontmatter(md: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", md, count=1, flags=re.DOTALL).strip()


def _summary_prompt(ocr_md: str, metadata: dict) -> str:
    title = metadata.get("title", "Unknown title")
    authors = ", ".join(metadata.get("authors") or [])
    year = metadata.get("year") or "unknown"
    venue = metadata.get("venue") or "unknown"

    return (
        f"You are summarizing a research paper for the Cardio Wiki — an interventional "
        f"cardiology intelligence system covering practice-changing trials, guidelines, "
        f"PCI techniques, structural heart interventions, and India-specific practice.\n\n"
        f"Paper: {title}\n"
        f"Authors: {authors}\n"
        f"Year: {year} · Venue: {venue}\n\n"
        f"First decide the article type from the OCR: completed trial/results paper, "
        f"study protocol/design paper, guideline/consensus, observational study, review, "
        f"or other. This classification controls the summary. If the paper is a protocol "
        f"or design/rationale paper, say clearly that no outcome results are reported.\n\n"
        f"Produce a Markdown summary with these sections:\n"
        f"- Paper Type (one sentence; include protocol/design status when applicable)\n"
        f"- Clinical Question (1 sentence)\n"
        f"- PICO (table: Population, Intervention, Comparator, Outcome)\n"
        f"- Key Results (only actual reported outcome results; if none, write "
        f"\"No outcome results are reported in this protocol\" and put sample-size "
        f"assumptions/planned endpoints in a separate paragraph called Design Assumptions)\n"
        f"- What Changed? (practice impact; for protocols, say no practice change yet)\n"
        f"- What Did Not Change? (things not proven)\n"
        f"- How I Would Explain This to a Patient (plain language)\n"
        f"- Relevance for Indian Practice (cost, availability, follow-up)\n"
        f"- Connections (use wikilinks like [[../../concepts/foo|Foo]] only for "
        f"concepts directly supported by the paper. Do not add coronary artery disease, "
        f"acute coronary syndrome, PCI, antiplatelet therapy, or heart failure links "
        f"unless the paper is actually about those topics.)\n\n"
        f"Accuracy rules:\n"
        f"- Do not infer completed trial findings from planned endpoints, hypotheses, "
        f"power calculations, expected event rates, or sample-size assumptions.\n"
        f"- Do not describe estimated rates used for power calculations as results.\n"
        f"- Do not claim practice change for protocols, rationale/design papers, or trials "
        f"without published outcome results.\n"
        f"- Preserve exact trial design details, eligibility criteria, endpoint definitions, "
        f"follow-up schedule, and sample size when they are stated in the OCR.\n"
        f"- If metadata title looks generic (for example \"Open access\" or \"Protocol\"), "
        f"use the actual full paper title from the OCR header.\n"
        f"- Be clinically precise. Quote exact numbers only when the paper states them. "
        f"Do not invent results or topic tags. The full OCR follows.\n\n"
        f"--- OCR ---\n{ocr_md[:120000]}"
    )


def gpt55_summarize_stream(ocr_md: str, metadata: dict) -> Generator[Tuple[str, str, int], None, None]:
    """Yield GPT-5.5 summary tokens as ("token", text, total_chars), then ("done", full_text, total_chars)."""
    key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OPENAI_SUMMARY_MODEL") or os.environ.get("OPENAI_INGEST_MODEL") or "gpt-5.5"
    fallback = os.environ.get("OPENAI_INGEST_FALLBACK", "gpt-4o")
    read_timeout = int(os.environ.get("OPENAI_STREAM_READ_TIMEOUT", "8"))
    idle_timeout = int(os.environ.get("OPENAI_STREAM_IDLE_TIMEOUT", "8"))
    max_stream_chars = int(os.environ.get("OPENAI_SUMMARY_MAX_STREAM_CHARS", "5000"))
    prompt = _summary_prompt(ocr_md, metadata)
    attempts = [
        (model, "max_completion_tokens"),
        (model, "max_tokens"),
    ]
    if fallback and fallback != model:
        attempts.append((fallback, "max_completion_tokens"))

    last_error = ""
    for attempt_model, token_field in attempts:
        payload = {
            "model": attempt_model,
            "messages": [{"role": "user", "content": prompt}],
            token_field: 8000,
            "stream": True,
        }
        full_text = ""
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=(30, read_timeout),
                stream=True,
            )
            if resp.status_code >= 400:
                last_error = resp.text[:1000]
                continue

            last_token_at = time.monotonic()
            for line in resp.iter_lines(decode_unicode=True):
                if full_text and time.monotonic() - last_token_at >= idle_timeout:
                    break
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                token = choice.get("delta", {}).get("content") or ""
                if token:
                    full_text += token
                    last_token_at = time.monotonic()
                    yield ("token", token, len(full_text))
                if choice.get("finish_reason") or len(full_text) >= max_stream_chars:
                    break

            if full_text.strip():
                yield ("done", full_text, len(full_text))
                return
            last_error = "OpenAI returned an empty structured summary."
        except Exception as exc:
            if full_text.strip():
                yield ("done", full_text, len(full_text))
                return
            last_error = str(exc)

    raise RuntimeError(f"GPT-5.5 structured summary failed: {last_error}")


# ---------------------------------------------------------------------------
# 5. GPT-5.5 final integration (with structured-summary fallback)
# ---------------------------------------------------------------------------


def _slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", (title or "untitled").lower())
    s = re.sub(r"\s+", "-", s).strip("-")
    return (s or "paper")[:80]


def _build_frontmatter(metadata: dict, slug: str, source_info: dict, body: str = "",
                       group_hint: str | None = None) -> str:
    title = metadata.get("title") or slug
    authors = metadata.get("authors") or []
    year = metadata.get("year") or "null"
    venue = metadata.get("venue") or "null"
    citations = metadata.get("citations") or 0
    arxiv_id = source_info.get("id") if source_info.get("kind") == "arxiv" else None
    doi = source_info.get("doi") or (metadata.get("external_ids") or {}).get("DOI")
    pmcid = source_info.get("pmcid") or (source_info.get("id") if source_info.get("kind") == "pmc" else None)
    abs_url = source_info.get("abs_url")
    today = time.strftime("%Y-%m-%d")
    tags = auto_tag(title, body)
    paper_type = infer_paper_type(title, body)
    domain = infer_domain(title, body, group_hint)
    if paper_type and paper_type not in tags:
        tags.append(paper_type)
    group_to_tag = {
        "coronary-pci": "pci",
        "acs": "acs",
        "imaging-guided-pci": "imaging-guided-pci",
        "physiology-guided-pci": "physiology-guided-pci",
        "antithrombotic": "antithrombotic",
        "structural-heart": "structural-heart",
        "valve-rheumatic": "rheumatic-heart-disease",
        "peripheral-endovascular": "peripheral-endovascular",
        "heart-failure-shock": "heart-failure",
        "prevention-lipids": "prevention",
        "ep-af": "atrial-fibrillation",
        "device-technology": "device-technology",
        "complications-safety": "complications-safety",
        "guidelines-appropriate-use": "guideline",
        "india-lmic": "india-practice",
    }
    hinted_tag = group_to_tag.get(group_hint or "")
    if hinted_tag and hinted_tag not in tags:
        tags.append(hinted_tag)

    # Compute the canonical "open original" URL: prefer DOI > PMC > abs_url > pdf_url
    source_url = None
    if doi:
        source_url = f"https://doi.org/{doi}"
    elif pmcid:
        source_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    elif abs_url:
        source_url = abs_url
    elif arxiv_id:
        source_url = f"https://arxiv.org/abs/{arxiv_id}"

    lines = [
        "---",
        f'title: "{title}"',
        "type: paper",
        "status: active",
        f"updated: {today}",
        f"year: {year}",
        f"paper_type: {paper_type}",
        f"domain: {domain}",
    ]
    if venue and venue != "null":
        lines.append(f'venue: "{venue}"')
    if arxiv_id:
        lines.append(f'arxiv_id: "{arxiv_id}"')
    if doi:
        lines.append(f'doi: "{doi}"')
    if pmcid:
        lines.append(f'pmcid: "{pmcid}"')
    if source_url:
        lines.append(f'source_url: "{source_url}"')
    if authors:
        lines.append("authors:")
        for a in authors:
            lines.append(f"  - {a}")
    lines.append("tags:")
    for t in tags:
        lines.append(f"  - {t}")
    citations_val = "null" if citations in (0, None) else str(citations)
    lines.append(f"citations: {citations_val}")
    lines.append("clinical_review: false")
    lines.append("sources:")
    lines.append(f'  ocr: ".grounding/md_fc/{slug}.md"')
    lines.append("  llm_drafted_by: gpt-5.5")
    lines.append("  llm_reviewed_by: gpt-5.5")
    lines.append(f'ingest_date: "{today}"')
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def normalize_page_body(body: str, title: str) -> str:
    """Fix common LLM formatting drift before writing Markdown."""
    clean = body.lstrip()
    title_heading = f"# {title}"
    if clean.startswith("# Summary for Cardio Wiki"):
        clean = re.sub(r"^# Summary for Cardio Wiki\s*", f"{title_heading}\n\n", clean, count=1)
    elif clean.startswith("# Summary"):
        clean = re.sub(r"^# Summary[^\n]*\s*", f"{title_heading}\n\n", clean, count=1)
    elif not clean.startswith("# "):
        clean = f"{title_heading}\n\n{clean}"

    if re.search(r"## PICO\s*\n\s*Population Intervention Comparator Outcome\s*\n", clean):
        clean = re.sub(
            r"## PICO\s*\n\s*Population Intervention Comparator Outcome\s*\n"
            r"(.+?)\s+Bi-atrial ablation\s+Left atrial ablation\s+(.+?)(?=\n\s*##|\Z)",
            "## PICO\n\n"
            "| Component | Description |\n"
            "|---|---|\n"
            r"| Population | \1 |\n"
            "| Intervention | Bi-atrial ablation |\n"
            "| Comparator | Left atrial ablation |\n"
            r"| Outcome | \2 |\n",
            clean,
            flags=re.DOTALL,
        )
    return clean.strip() + "\n"


def infer_paper_type(title: str, body: str) -> str:
    text = f"{title}\n{body[:4000]}".lower()
    if "study protocol" in text or "protocol/design" in text or "no outcome results are reported in this protocol" in text:
        return "protocol"
    if "meta-analysis" in text or "systematic review" in text:
        return "meta-analysis"
    if "registry" in text:
        return "registry"
    if "guideline" in text:
        return "guideline"
    if "randomised controlled trial" in text or "randomized controlled trial" in text or " rct" in text:
        return "trial"
    if "review" in text:
        return "review"
    return "trial"


def gpt55_integrate(structured_summary: str, metadata: dict, source_info: dict, slug: str) -> str:
    """Final integration step. Uses GPT-5.5 to polish the structured summary into a wiki page."""
    title = metadata.get("title") or slug
    abs_url = source_info.get("abs_url") or source_info.get("pdf_url")

    system = (
        "You are the editor for the Cardio Wiki, an interventional cardiology intelligence system. "
        "Take the structured summary below and produce a clean, Markdown-only wiki page body "
        "(NO YAML frontmatter — that will be prepended separately). "
        "Start with a level-1 heading using the paper title, then sections: "
        "Paper Type, Clinical Question, PICO (table), Key Results, What Changed?, "
        "What Did Not Change?, How I Would Explain This to a Patient, "
        "Relevance for Indian Practice, Related Trials. "
        "If the source is a protocol/design paper, preserve the statement that no outcome "
        "results are reported, keep planned endpoints and sample-size assumptions separate "
        "from results, and state that no practice change should be inferred yet. "
        "Do not add unrelated coronary/PCI/ACS/antiplatelet content unless present in the "
        "summary. Be clinically precise; do not invent claims. Keep paragraphs tight."
    )
    user = (
        f"Title: {title}\nLink: {abs_url}\n\n"
        f"--- Structured summary ---\n{structured_summary}"
    )

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return structured_summary

    model = os.environ.get("OPENAI_INTEGRATE_MODEL", "gpt-5.5")
    fallback = os.environ.get("OPENAI_INTEGRATE_FALLBACK", "gpt-4o")
    timeout = int(os.environ.get("OPENAI_INTEGRATE_TIMEOUT", "90"))

    for attempt_model in [model, fallback] if fallback and fallback != model else [model]:
        for token_field in ["max_completion_tokens", "max_tokens"]:
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": attempt_model,
                        token_field: 6000,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.2,
                    },
                    timeout=(10, timeout),
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    if content and content.strip():
                        return content
                print(f"[gpt55_integrate] {attempt_model} returned status {resp.status_code}: {resp.text[:200]}")
            except requests.exceptions.Timeout:
                print(f"[gpt55_integrate] {attempt_model} timed out after {timeout}s")
            except Exception as exc:
                print(f"[gpt55_integrate] {attempt_model} failed: {type(exc).__name__}: {exc}")

    return structured_summary


# ---------------------------------------------------------------------------
# 6. Git commit + push for /add-paper
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> Tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def _github_repo_slug() -> str | None:
    env = os.environ.get("GITHUB_REPO")
    if env and "/" in env:
        return env.strip().strip("/")

    owner = os.environ.get("RAILWAY_GIT_REPO_OWNER")
    name = os.environ.get("RAILWAY_GIT_REPO_NAME")
    if owner and name:
        return f"{owner}/{name}"

    rc, origin_url = _git(["remote", "get-url", "origin"], BASE_DIR)
    if rc == 0:
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+)(?:\.git)?", origin_url.strip())
        if m:
            return m.group(1)
    return None


def _push_via_contents_api(
    *,
    token: str,
    repo: str,
    branch: str,
    files: list[tuple[str, bytes]],
    commit_msg: str,
    author_name: str,
    author_email: str,
) -> dict:
    import base64

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "cardio-wiki-bot",
    }
    probe_url = f"https://api.github.com/repos/{repo}"
    try:
        probe = requests.get(probe_url, headers=headers, timeout=20)
    except requests.RequestException as exc:
        return {"ok": False, "stage": "contents-api", "output": f"Repo probe failed: {exc}"}
    if probe.status_code != 200:
        return {
            "ok": False,
            "stage": "contents-api",
            "output": f"Repo probe GET {probe_url} -> HTTP {probe.status_code}: {probe.text[:300]}",
        }

    default_branch = (probe.json() or {}).get("default_branch")
    if default_branch and default_branch != branch and not os.environ.get("GIT_BRANCH"):
        branch = default_branch

    base_url = f"https://api.github.com/repos/{repo}/contents"
    last_sha = None
    for rel_path, content in files:
        existing_sha = None
        get_url = f"{base_url}/{rel_path}?ref={branch}"
        try:
            existing = requests.get(get_url, headers=headers, timeout=30)
            if existing.status_code == 200:
                existing_sha = (existing.json() or {}).get("sha")
        except requests.RequestException:
            existing_sha = None

        payload = {
            "message": commit_msg,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": branch,
            "committer": {"name": author_name, "email": author_email},
            "author": {"name": author_name, "email": author_email},
        }
        if existing_sha:
            payload["sha"] = existing_sha
        try:
            updated = requests.put(f"{base_url}/{rel_path}", headers=headers, json=payload, timeout=60)
        except requests.RequestException as exc:
            return {"ok": False, "stage": "contents-api", "output": f"PUT {rel_path}: {exc}"}
        if updated.status_code not in (200, 201):
            return {
                "ok": False,
                "stage": "contents-api",
                "output": f"PUT {rel_path} -> HTTP {updated.status_code}: {updated.text[:400]}",
            }
        last_sha = ((updated.json() or {}).get("commit") or {}).get("sha")
    return {"ok": True, "sha": last_sha[:7] if last_sha else None, "via": "contents-api", "branch": branch}


def commit_and_push_paper(slug: str, title: str, page_path: Path, ocr_path: Path) -> dict:
    if os.environ.get("GIT_AUTOPUSH", "1") == "0":
        return {"skipped": True, "reason": "GIT_AUTOPUSH=0"}

    author_name = os.environ.get("GIT_AUTHOR_NAME") or "Cardio Wiki Bot"
    author_email = os.environ.get("GIT_AUTHOR_EMAIL") or "cardio-wiki-bot@users.noreply.github.com"
    branch = os.environ.get("GIT_BRANCH") or "main"
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    rel_page = page_path.relative_to(BASE_DIR).as_posix()
    rel_ocr = ocr_path.relative_to(BASE_DIR).as_posix()

    with _GIT_LOCK:
        rc, _ = _git(["rev-parse", "--git-dir"], BASE_DIR)
        if rc == 0:
            rc, remotes = _git(["remote", "-v"], BASE_DIR)
            if rc == 0 and "origin" in remotes:
                rc, out = _git(["add", "--", rel_page, rel_ocr], BASE_DIR)
                if rc != 0:
                    return {"ok": False, "stage": "add", "output": out}
                rc, _ = _git(["diff", "--cached", "--quiet"], BASE_DIR)
                if rc == 0:
                    return {"skipped": True, "reason": "no staged changes"}
                commit_msg = (
                    f"Add paper via /add-paper: {title}\n\n"
                    f"slug: {slug}\nAuto-committed by the Cardio Wiki ingest pipeline."
                )
                rc, out = _git(
                    ["-c", f"user.name={author_name}", "-c", f"user.email={author_email}",
                     "commit", "-m", commit_msg],
                    BASE_DIR,
                )
                if rc != 0:
                    return {"ok": False, "stage": "commit", "output": out}
                _, sha = _git(["rev-parse", "--short", "HEAD"], BASE_DIR)
                sha = sha.strip()
                _, origin_url = _git(["remote", "get-url", "origin"], BASE_DIR)
                origin_url = origin_url.strip()
                if token and origin_url.startswith("https://"):
                    auth_url = re.sub(r"^https://([^@]+@)?", f"https://x-access-token:{token}@", origin_url)
                    rc, out = _git(["push", auth_url, f"HEAD:{branch}"], BASE_DIR)
                else:
                    rc, out = _git(["push", "origin", f"HEAD:{branch}"], BASE_DIR)
                if rc != 0:
                    return {"ok": False, "stage": "push", "sha": sha, "output": out}
                return {"ok": True, "sha": sha, "via": "git"}

        if not token:
            return {"skipped": True, "reason": "no .git/ and GITHUB_TOKEN not set"}
        repo = _github_repo_slug()
        if not repo:
            return {"skipped": True, "reason": "GITHUB_REPO not set"}
        return _push_via_contents_api(
            token=token,
            repo=repo,
            branch=branch,
            files=[(rel_page, page_path.read_bytes()), (rel_ocr, ocr_path.read_bytes())],
            commit_msg=f"Add paper via /add-paper: {title}\n\nslug: {slug}",
            author_name=author_name,
            author_email=author_email,
        )


# ---------------------------------------------------------------------------
# 7. Pipeline driver (yields events)
# ---------------------------------------------------------------------------


def _event(name: str, **payload) -> Tuple[str, dict]:
    return (name, payload)


def ingest_pipeline(
    *,
    url: str | None = None,
    pdf_bytes: bytes | None = None,
    title_hint: str | None = None,
    slug_hint: str | None = None,
    source_url: str | None = None,
    group_hint: str | None = None,
    do_citation: bool = True,
    do_openai: bool = True,
    do_autopush: bool = True,
) -> Generator[Tuple[str, dict], None, None]:
    try:
        # ── 1. Detect / acquire ────────────────────────────────────────────
        if pdf_bytes:
            yield _event("status", stage="source", message=f"Using uploaded PDF ({len(pdf_bytes)} bytes)")
            source_info = {"kind": "upload", "id": None, "pdf_url": None,
                           "abs_url": source_url or None}
            if source_url:
                pmc_m = PMCID_IN_OCR_RE.search(source_url)
                if pmc_m:
                    source_info["kind"] = "pmc"
                    source_info["id"] = pmc_m.group(1)
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

        # Extract DOI / PMC ID from OCR for citation links
        ocr_doi = extract_doi_from_ocr(ocr_md)
        ocr_pmcid = extract_pmcid_from_ocr(ocr_md)
        if ocr_doi:
            source_info["doi"] = ocr_doi
            yield _event("status", stage="ocr", message=f"DOI from OCR: {ocr_doi}")
        if ocr_pmcid and not source_info.get("id"):
            source_info["pmcid"] = ocr_pmcid
            yield _event("status", stage="ocr", message=f"PMC ID from OCR: {ocr_pmcid}")
        elif ocr_pmcid:
            source_info["pmcid"] = ocr_pmcid

        ocr_title = title_hint
        if not ocr_title:
            ocr_title = extract_title_from_ocr(ocr_md)

        # ── 3. Citation lookup ────────────────────────────────────────────
        metadata = None
        if do_citation:
            yield _event("status", stage="citation", message="Semantic Scholar — looking up metadata + citations…")
            if source_info.get("kind") == "arxiv":
                metadata = semantic_scholar_lookup(arxiv_id=source_info["id"])
            if not metadata and source_info.get("kind") == "doi":
                metadata = semantic_scholar_lookup(doi=source_info["id"])
            if not metadata and source_info.get("doi"):
                metadata = semantic_scholar_lookup(doi=source_info["doi"])
            if not metadata and ocr_title:
                metadata = semantic_scholar_lookup(title=ocr_title)
        else:
            yield _event("status", stage="citation", message="Citation lookup skipped")
        if not metadata:
            metadata = {"title": ocr_title or "Untitled paper", "authors": [], "year": None,
                        "venue": None, "citations": 0, "abstract": "", "external_ids": {}}
            if do_citation:
                yield _event("status", stage="citation", message="Semantic Scholar miss — trying quick Grok citation estimate…")
                est = xai_citation_estimate(metadata["title"])
                if est is not None:
                    metadata["citations"] = est
                    yield _event("status", stage="citation", message=f"Grok estimate: ~{est} citations")
                else:
                    yield _event("status", stage="citation", message="Grok citation estimate unavailable — continuing")
        else:
            yield _event("status", stage="citation",
                         message=f"Found: {metadata.get('title','?')[:80]} · {metadata.get('citations',0)} citations")

        # Backfill year/venue from OCR header, then Grok, if Semantic Scholar didn't have them
        if not metadata.get("year") or not metadata.get("venue"):
            ocr_meta = extract_metadata_from_ocr(ocr_md)
            if not metadata.get("year") and ocr_meta.get("year"):
                metadata["year"] = ocr_meta["year"]
                yield _event("status", stage="citation",
                             message=f"Year from OCR header: {ocr_meta['year']}")
            if not metadata.get("venue") and ocr_meta.get("venue"):
                metadata["venue"] = ocr_meta["venue"]
                yield _event("status", stage="citation",
                             message=f"Venue from OCR header: {ocr_meta['venue']}")
        if not metadata.get("year") and metadata.get("title"):
            yr = xai_year_estimate(metadata["title"])
            if yr:
                metadata["year"] = yr
                yield _event("status", stage="citation",
                             message=f"Year from Grok: {yr}")

        yield _event("metadata", **{k: v for k, v in metadata.items() if k != "abstract"})

        # ── 4. GPT-5.5 streaming summarization ───────────────────────────
        yield _event("status", stage="summary", message="GPT-5.5 — streaming structured summary (this takes 1-3 min)…")
        structured_summary = ""
        last_summary_emit = 0
        for kind, payload, total_chars in gpt55_summarize_stream(ocr_md, metadata):
            if kind == "token":
                if total_chars - last_summary_emit >= 500:
                    yield _event("summary_progress", chars=total_chars)
                    last_summary_emit = total_chars
            elif kind == "done":
                structured_summary = payload
                yield _event("summary_progress", chars=total_chars)
        yield _event("status", stage="summary", message=f"GPT-5.5 summary done — {len(structured_summary):,} chars")

        # ── 5. Final integration via GPT-5.5 ─────────────────────────────
        slug = slug_hint or _slugify(metadata.get("title") or ocr_title or "paper")
        if do_openai:
            yield _event("status", stage="integrate", message="Final integration via GPT-5.5…")
            body = gpt55_integrate(structured_summary, metadata, source_info, slug)
            if body == structured_summary:
                yield _event("status", stage="integrate", message="GPT-5.5 integration unavailable — using raw structured summary")
            body = normalize_page_body(body, metadata.get("title") or slug)
            yield _event("status", stage="integrate", message=f"Integration done — {len(body):,} chars")
        else:
            yield _event("status", stage="integrate", message="Final integration skipped — using raw structured summary")
            body = normalize_page_body(structured_summary, metadata.get("title") or slug)

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

        full_page = _build_frontmatter(metadata, slug, source_info, body=body, group_hint=group_hint) + body.lstrip("\n")
        page_path.write_text(full_page, encoding="utf-8")

        yield _event("status", stage="write", message=f"Wrote {page_path.relative_to(BASE_DIR)}")
        git_result = {"skipped": True, "reason": "disabled"}
        if do_autopush:
            yield _event("status", stage="publish", message="Committing and pushing paper files…")
            git_result = commit_and_push_paper(slug, metadata.get("title") or slug, page_path, ocr_path)
            if git_result.get("ok"):
                yield _event("status", stage="publish", message=f"Pushed commit {git_result.get('sha')}")
            elif git_result.get("skipped"):
                yield _event("status", stage="publish", message=f"Push skipped: {git_result.get('reason')}")
            else:
                yield _event("status", stage="publish", message=f"Push failed at {git_result.get('stage')}: {git_result.get('output', '')[:160]}")
        else:
            yield _event("status", stage="publish", message="Auto-push skipped")
        yield _event("done", slug=slug, page_path=str(page_path.relative_to(BASE_DIR)),
                     ocr_path=str(ocr_path.relative_to(BASE_DIR)),
                     citations=metadata.get("citations", 0),
                     title=metadata.get("title") or slug,
                     git_status=("pushed" if git_result.get("ok") else "skipped" if git_result.get("skipped") else "failed"),
                     git_sha=git_result.get("sha"))
    except Exception as exc:
        yield _event("error", message=f"{type(exc).__name__}: {exc}")
