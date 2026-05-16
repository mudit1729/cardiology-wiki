import unittest

from paper_ingest import auto_tag, extract_title_from_ocr, infer_domain, infer_paper_type, normalize_page_body, _summary_prompt


class PaperIngestPromptTests(unittest.TestCase):
    def test_extract_title_skips_bmj_boilerplate(self):
        ocr = """Open access

Protocol

# BMJ Open Bi-atrial versus left atrial ablation for patients with rheumatic mitral valve disease and non-paroxysmal atrial fibrillation (ABLATION): rationale, design and study protocol for a multicentre randomised controlled trial

Chunyu Yu, Haojie Li, Yang Wang

To cite: Yu C, Li H, Wang Y, et al. Bi-atrial versus left atrial ablation for patients with rheumatic mitral valve disease and non-paroxysmal atrial fibrillation (ABLATION): rationale, design and study protocol for a multicentre randomised controlled trial. BMJ Open 2022;12:e064861. doi:10.1136/bmjopen-2022-064861
"""
        self.assertEqual(
            extract_title_from_ocr(ocr),
            "Bi-atrial versus left atrial ablation for patients with rheumatic mitral valve disease and non-paroxysmal atrial fibrillation (ABLATION): rationale, design and study protocol for a multicentre randomised controlled trial",
        )

    def test_summary_prompt_separates_protocol_assumptions_from_results(self):
        prompt = _summary_prompt(
            "Methods and analysis The ABLATION trial is a prospective protocol.",
            {"title": "Open access", "authors": [], "year": 2022, "venue": "BMJ Open"},
        )
        self.assertIn("study protocol/design paper", prompt)
        self.assertIn("No outcome results are reported in this protocol", prompt)
        self.assertIn("Do not infer completed trial findings", prompt)
        self.assertIn("Do not describe estimated rates used for power calculations as results", prompt)
        self.assertIn("metadata title looks generic", prompt)

    def test_rheumatic_af_surgery_is_not_tagged_af_pci(self):
        tags = auto_tag(
            "Bi-atrial versus left atrial ablation for rheumatic mitral valve disease and non-paroxysmal atrial fibrillation",
            "warfarin anticoagulation during mitral valve surgery for atrial fibrillation",
        )
        self.assertIn("atrial-fibrillation", tags)
        self.assertIn("mitral", tags)
        self.assertIn("rheumatic", tags)
        self.assertNotIn("af-pci", tags)

    def test_normalize_page_body_replaces_generic_heading_and_pico(self):
        body = """# Summary for Cardio Wiki

## PICO

Population Intervention Comparator Outcome
Patients with rheumatic mitral valve disease and non-paroxysmal atrial fibrillation undergoing mitral valve surgery Bi-atrial ablation Left atrial ablation Freedom from atrial tachyarrhythmias at 12 months

## Key Results

No outcome results are reported in this protocol.
"""
        normalized = normalize_page_body(body, "Full Paper Title")
        self.assertTrue(normalized.startswith("# Full Paper Title"))
        self.assertIn("| Component | Description |", normalized)
        self.assertIn("| Intervention | Bi-atrial ablation |", normalized)
        self.assertNotIn("Population Intervention Comparator Outcome", normalized)

    def test_protocol_type_is_inferred_from_summary(self):
        self.assertEqual(
            infer_paper_type("ABLATION protocol", "No outcome results are reported in this protocol."),
            "protocol",
        )

    def test_domain_is_inferred_for_valve_rheumatic_paper(self):
        self.assertEqual(
            infer_domain(
                "Bi-atrial versus left atrial ablation for rheumatic mitral valve disease",
                "Patients with rheumatic mitral valve disease and non-paroxysmal atrial fibrillation underwent mitral valve surgery.",
            ),
            "valve-rheumatic",
        )

    def test_domain_hint_overrides_inference(self):
        self.assertEqual(
            infer_domain("General coronary paper", "PCI and stent optimization.", "device-technology"),
            "device-technology",
        )


if __name__ == "__main__":
    unittest.main()
