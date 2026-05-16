import unittest

from paper_ingest import extract_title_from_ocr, _summary_prompt


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


if __name__ == "__main__":
    unittest.main()
