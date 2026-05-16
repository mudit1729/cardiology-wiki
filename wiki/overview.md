---
title: Overview
type: concept
status: active
tags:
  - overview
  - navigation
updated: 2026-05-15
---

# Cardiology Wiki Overview

This wiki is an early ingestion workspace for source-grounded summaries of cardiology
literature. It is **not** yet a complete cardiology guideline / trial / procedure
encyclopedia, and it should not be used as a clinical reference.

## Current State (2026-05-15)

The live wiki contains:

- **6 ingested paper summaries** under `wiki/sources/papers/` — produced by the
  Mistral OCR &rarr; DeepSeek summary &rarr; XAI Grok integration pipeline from
  open-access PDFs in PubMed Central.
- **Taxonomy / planning pages** under `wiki/taxonomies/research-map.md` and `wiki/overview.md`.
- The `wiki/trials/`, `wiki/guidelines/`, `wiki/procedures/`, `wiki/conferences/`,
  and `wiki/concepts/` directories are **empty**. They are placeholders for
  future ingestion or hand-authored content.

LLM-generated summaries on each paper page have **not** been clinically reviewed.
Each page carries `clinical_review: false` in its frontmatter.

## How Pages Are Created

New paper summaries are added through `/add-paper`. The pipeline:

1. **Mistral OCR** extracts text from the PDF
2. **Semantic Scholar** fetches metadata and citation counts (XAI Grok fallback for citation estimate)
3. **Year, venue, DOI, PMC ID** are extracted from the OCR header if Semantic Scholar misses
4. **DeepSeek** drafts a structured summary (Clinical Question, PICO, Key Results, etc.)
5. **XAI Grok** integrates the summary into a final wiki page
6. Auto-generated **tags** are derived from a controlled keyword vocabulary (conservative: title or repeated body match required, exclusion-criteria contexts filtered)

Each page records `source_url`, `doi`, `pmcid`, and `arxiv_id` (when available)
so the original paper is one click away.

## Views

- **Trials** — filterable grid of all paper / trial / guideline / procedure / conference pages
- **Graph** — interactive force-directed network showing connections between papers
- **Timeline** — horizontal timeline of papers grouped by research direction
- **Tags** — tag cloud with filtered results
- **Chat** — RAG-powered Q&A using selected paper summaries as context (DeepSeek streaming)
- **Search** — full-text search across the entire wiki

## Roadmap

- Expand the source-grounded paper corpus by ingesting more open-access cardiology RCTs
- Build out the empty `wiki/trials/`, `wiki/guidelines/`, `wiki/procedures/`,
  `wiki/conferences/`, `wiki/concepts/` directories with reviewed content
- Add clinician-reviewed pages with `clinical_review: true` once such review exists
