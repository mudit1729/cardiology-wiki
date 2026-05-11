---
title: Overview
type: concept
status: active
tags:
  - overview
  - navigation
updated: 2026-05-10
---

# Cardiology Wiki Overview

An interventional cardiology intelligence system covering practice-changing trials, current guidelines, procedural techniques, and conference updates.

## What This Wiki Covers

### Trials
Practice-changing randomized controlled trials in interventional cardiology organized by domain: stable CAD, ACS/STEMI, imaging-guided PCI, physiology-guided PCI, antiplatelet therapy, AF+PCI, TAVR/structural heart, lipid/prevention, and heart failure.

Each trial page includes a structured summary with clinical question, PICO table, key results, what changed in practice, what did not change, a one-minute patient explanation, India practice relevance, and connections to related trials and guidelines.

### Guidelines
Major society guidelines from ACC/AHA, ESC, and SCAI with class of recommendation tables, key updates from prior versions, and practical implementation notes.

### Procedures
Procedural playbooks for common interventional cardiology procedures including radial access PCI, femoral access and closure, IVUS-guided stenting, FFR/iFR measurement, rotational atherectomy, chronic total occlusion PCI, TAVR, and mechanical circulatory support.

### Conferences
Intelligence from major cardiology conferences including ACC, AHA, TCT, EuroPCR, and ESC with late-breaking trial summaries and practice-changing takeaways.

### Concepts
Foundational concept pages that tie together trials, guidelines, and procedures within each domain.

## How It's Organized

Pages are organized in a directory structure:

- `wiki/trials/` — trial summaries grouped by domain
- `wiki/guidelines/` — guideline summaries
- `wiki/procedures/` — procedural playbooks
- `wiki/conferences/` — conference intelligence
- `wiki/concepts/` — concept overview pages
- `wiki/syntheses/` — cross-domain synthesis pages
- `wiki/sources/papers/` — ingested paper summaries

## Pipeline

New papers can be ingested through the **Add** page. The pipeline:
1. **Mistral OCR** extracts text from the PDF
2. **Semantic Scholar** fetches metadata and citation counts (XAI Grok fallback)
3. **DeepSeek** generates a structured cardiology-focused summary
4. **XAI Grok** performs final integration and writes the wiki page

## Views

- **Trials** — filterable grid of all trial/evidence pages
- **Graph** — interactive force-directed network showing connections between trials
- **Timeline** — horizontal timeline of trials grouped by research direction
- **Tags** — tag cloud with filtered results
- **Chat** — RAG-powered Q&A using selected trial summaries as context
- **Search** — full-text search across the entire wiki
