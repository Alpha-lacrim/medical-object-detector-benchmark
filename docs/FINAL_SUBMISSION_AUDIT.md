# Final submission audit and author-declaration follow-up

**2026-09-27 Batch 53 update:** The canonical Markdown now includes a verified
[secondary YOLO numerical-path sensitivity](YOLO_NUMERICAL_PATH_SENSITIVITY_V1.md).
Historical primary/external evidence remains unchanged. The declaration-wording
follow-up below refreshes the PDF and numerical trace to include this sensitivity.
Current verification has 427 numerical bindings/36 semantic guards and 88
publication artifacts; the experiment retains its own protocol, provenance and
complete metric replay. Final author review is still required before submission.

## 2026-09-27 declaration wording and PDF refresh

At the author's request, two Section 8 drafting instructions were replaced with
plain statements of the pending ethics-review and submission-version decisions.
No approval, exemption, consent waiver, public submission identifier or DOI was
invented. The author's subsequent edit makes patient and public involvement
the final declaration and shortens the submission-version statement for the
requested repository publication. Scientific claims, results and their source
bindings are unchanged.

The canonical PDF was rebuilt with the documented Pandoc/pdfLaTeX command and
the numerical trace now contains all 427 verified bindings. All 16 pages were
rendered with Poppler and visually inspected, with a separate inspection of
the revised declarations on page 15. The Batch 53 sensitivity appears on page
13. Tables, figures, references and page boundaries remain legible, with no
clipping, overlap or missing glyphs. The automated QA receipt records 29 internal
links, 33 URI targets and zero errors; the final TeX log has no overflow,
undefined-reference or missing-character warnings.

- Manuscript SHA-256: `07ff736eb9c25a6b0178a67e7bafc8a6ceff75797ef45bbe30789a69a960c27c`.
- PDF SHA-256: `fce7ac64779ef068b218ff0fcc66bd2bb6c9594566bfe08efbfb6a5551497f31`.
- [Build receipt](../report/paper_build_manifest.json) binds current inputs/output;
  [QA receipt](../report/paper_pdf_qa.json) records the rendered file checks.
- Claim verification: 427 numerical claims and all 36 semantic guards pass.
  Scientific artifact verification: 88 artifacts, 646 present input bindings,
  212 result references, zero unavailable inputs.
- Repository support links are pinned to local commit
  `490c55794e2bfc163746089ffb44e7f195ef534b`, which contains the supporting files.
  The author explicitly authorized publishing the pending manuscript/PDF,
  author details and study artifacts to the configured GitHub main branch.
  This supporting revision is not a submission-release selection.

The two decisions remain tracked in [the author action record](AUTHOR_DECLARATIONS_TODO.md)
and reporting checklist. No new inference or training occurred. The prior
PDF/receipts are retained in Git history; the archived
Batch 52 scientific review files remain unchanged. Checks and render previews
for the wording edit are under local `tmp/pdfs/declaration-wording-20260927/`.
The final refresh is recorded under `tmp/pdfs/declarations-final-20260927/`:
all 15 other pages are pixel-identical to the preceding visual review, and the
changed declaration page was inspected separately.

## Historical author-review record: 2026-09-25

Updated 2026-09-25. **READY FOR CONTINUED AUTHOR REVIEW; NOT READY FOR JOURNAL
SUBMISSION.** Author-supplied facts are now incorporated. Institutional/journal
ethics and consent applicability, the exact submission release, journal selection
and final review by both authors remain open.

## Scientific audit and preserved evidence

The complete [2026-09-24 adversarial audit](FINAL_SUBMISSION_AUDIT_2026-09-24.md)
is preserved byte-for-byte. It contains the 51-value adversarial trace,
21-analysis scope table, exact-HEAD Ubuntu/Windows CI evidence, clean-checkout
boundary checks, authorized full-data replay, uncertainty/FROC/threshold/AMP/
repair/timing audits, source review and the original PDF inspection.
Its numerical values, scientific conclusions and verification results remain
applicable to the unchanged scientific evidence. Its author-action status,
source/PDF hashes and pagination describe the prior snapshot, superseded here.
The [September 1 audit](FINAL_SUBMISSION_AUDIT_2026-09-01.md) and
[prior manuscript audit](FINAL_MANUSCRIPT_AUDIT.md) remain historical records.

Exact prior source, PDF, trace, formatting filter and build/QA receipts are in
[the review archive](../report/provenance/batch52/). Its
[archive manifest](../report/provenance/batch52/archive_manifest.json) binds all
six files. Narrow Git attributes preserve their original bytes. The archive
is historical, not a second canonical manuscript or a new scientific freeze.

No method, cohort, checkpoint, threshold, result, confidence interval, scientific
configuration or scientific dependency changed in this follow-up. All five runs,
external absolute collapse and failed frozen-threshold transport remain visible.
The external YOLO upper-budget missing-support bound still permits reversal;
Faster R-CNN cap saturation remains an additional support limit. The precision
path caveat and distinction between training-procedure and fixed-checkpoint
uncertainty are unchanged.

### Repository and author state at 2026-09-25

Branch is `main`; HEAD remains
`ac6975df799a20c55d97204ddcdcbd0dab6675f9`. The index remains empty. The earlier
Batch 52 changes and these author edits are local and unstaged; no commit,
push, checkpoint release or journal submission was performed. The pre-existing
README wget link and unrelated files were preserved. CODEX/HANDOFF remain local.

`gh repo view Alpha-lacrim/medical-object-detector-benchmark --json
url,visibility,defaultBranchRef` verified the configured repository is public
with default branch `main` on 2026-09-25. A restricted-network attempt failed;
the authorized read-only retry succeeded. This is visibility verification,
not publication of the current working tree or selection of a submission revision.

| Declaration | Current record |
|---|---|
| Authors | Pouyan Delivandani, then Mohammad Amin Hajialirezaei; no additional authors currently |
| Affiliations | Computer Engineering, IKIU, Qazvin, Iran; Computer Engineering, K. N. Toosi University of Technology, Tehran, Iran, respectively |
| Correspondence | Pouyan; both user-supplied email addresses appear on the title page |
| Contributions | Pouyan: methodology, software, data curation, investigation, analysis, visualization, original draft, review/editing. Mohammad Amin: analysis, visualization, review/editing. No equal-contribution designation |
| Funding/support | No external funding, project-specific scholarship, institutional research support or sponsored computing/equipment; authors used their own resources |
| Competing interests | Pouyan confirmed neither author has financial or nonfinancial competing interests related to the work |
| Ethics/consent | No separate institutional approval, exemption or written determination was sought or obtained. No participants recruited and no new consent obtained. Source VinDr approval/waiver is attributed to its original study |
| Patient/public involvement | None in research question, design/conduct, interpretation or dissemination, as confirmed by Pouyan |
| Availability | Public repository, software license and source-provider routes stated. All ten trained checkpoints remain unpublished at initial submission by author decision; no promised later release or availability-on-request claim |

The [author fact/action record](AUTHOR_DECLARATIONS_TODO.md) distinguishes user
confirmations, externally documented facts and outstanding decisions. Original
VinDr ethics/consent facts were checked against the
[official release Methods](https://physionet.org/content/vindr-cxr/1.0.0/).
NIH de-identification/public release was checked against the
[official NIH announcement](https://irp.nih.gov/news-and-events/in-the-news/nih-clinical-center-provides-one-of-the-largest-publicly-available-chest-x).
Neither source establishes a determination for the authors' secondary analysis.
The existing VinDr release bibliography key supports the added ethics citations;
no new bibliography entry was needed. CLAIM item 44 is now Yes on supplied funding
facts. Item 43 remains incomplete pending the exact public submission revision.
The crosswalk remains a reporting aid, not certification.

## Remaining author/submission actions

- Establish applicable institutional and target-journal ethics/consent requirements.
  The stated absence of a determination is factual; it is not an exemption or waiver.
- Choose the target journal and satisfy its format, authorship declarations,
  competing-interest scope and official reporting-checklist requirements.
- Select and verify the exact public commit/release/archive for submission.
  No DOI is claimed. Keep raw/restricted data and unpublished checkpoint binaries
  outside that release, as declared.
- Complete final review by both authors. Future revisions/review have not been
  presented as already performed. Existing upstream-metadata, subgroup-reporting
  and cohort-flow gaps remain disclosed and are not repaired by declarations.

At that review, two explicit AUTHOR ACTION REQUIRED blocks remained in the manuscript:
ethics/consent applicability and the submission release identifier. The scientific
review status in the subtitle remains appropriate. This is not submission clearance.

### Verification and reproducibility at 2026-09-25

The [complete trace](SUBMISSION_NUMERICAL_TRACE.csv) was regenerated for current
manuscript line numbers. All 417 bindings preserve the prior manuscript values,
source locators/calculations, unrounded values, rounding and tolerances; only line
locations changed. The [preserved audit's 51-value selection](FINAL_SUBMISSION_AUDIT_2026-09-24.md#numerical-traceability)
and scope table still describe the same science.

Executed after the author edits (CPU means `.venv/Scripts/python.exe`):

| Command/check | Result |
|---|---|
| `CPU scripts/verify_paper_claims.py` | 417 numerical bindings and all 33 semantic guards PASS |
| `CPU -m scripts.export_paper_trace --manifest report/paper_claim_sources.yaml --output docs/SUBMISSION_NUMERICAL_TRACE.csv` | 417 current trace rows exported |
| `CPU scripts/verify_scientific_artifacts.py --manifest results/publication_artifact_manifest.json` | 83 artifacts, all 580 local input bindings, 201 result references PASS |
| `CPU -m scripts.verify_frozen_external --mode freeze` | Original 38 frozen bindings PASS |
| `CPU scripts/check_bibliography.py --bibliography report/references.bib --manuscripts report/paper_draft.md report/report.md report/Manuscript_FasterRCNN_vs_YOLO11s_LungOpacity.md` | 40 unique entries, 19/19/27 citation keys PASS |
| `python -m scripts.build_paper_pdf --config configs/paper_build.yaml` in the separate document environment | Canonical 16-page PDF regenerated |
| `python -m scripts.check_paper_pdf --pdf output/pdf/rsna_vindr_article.pdf --output report/paper_pdf_qa.json --text tmp/pdfs/batch52_article/author-declarations.txt` | 16 nonempty pages; 29 resolved internal links, 32 URI targets including both mailto addresses; zero errors |
| Source-body/trace preservation, archive hashes, local publication links, PDF input/output hashes and whitespace | PASS; final receipt under `tmp/author-actions/` |

The final local support check resolved 298 publication links and eight immutable
PDF support targets; rehashed all ten checkpoints (962,924,817 bytes) and their
configs; confirmed all binaries remain Git-ignored with no public download URL;
and resolved 44 README config paths and 28 modules/packages. All six archived
review files retain their hashes under Git's configured byte-preservation rules.

The original public-checkout 438-test and authorized 445-test runs, full ten-run
external metric/source replay and both 2,000-draw bootstrap replays were executed
on 2026-09-24 as documented in the preserved audit. They were not rerun for this
non-scientific author/formatting update. No new Python or scientific implementation
was introduced; the presentation-only Lua change renders author details, keeps
Table 2 together and starts the bibliography on a fresh page. Scientific artifact
hash verification and current claim checks above were rerun.

### PDF inspection at 2026-09-25

Every one of the 16 final pages was rendered with Poppler and visually inspected.
The title/author/affiliation/email block is on page 1; all seven tables and five
figure images are legible; declarations are on pages 14-15; all 19 references
fit together on page 16. Figure 2 (page 9) and Table 7 (page 12) retain their exact
support/reversal/cap caveats. Table 2 is kept whole on page 7 after the added
front matter; whitespace on page 6 avoids a one-row table continuation.
The separate bibliography page leaves whitespace after the declarations.
No clipped content, missing glyph, overlap, unresolved citation or unintended
placeholder remains. The two outstanding author-action blocks stay visible.
The final TeX log has no overflow, undefined-reference, missing-character or
warning messages. No scientific content was edited in the rendered PDF.

- Current manuscript SHA-256: `fc38d6e3a819acdcb05647a2a18e2b0abc5615b9f39f637dbb2cb2c83b959910`.
- Current PDF SHA-256: `4c783e10d4a64c9c29dd8e375b0c48db5f092c93689eb7508664858c815bd6b3`.
- [Build receipt](../report/paper_build_manifest.json) binds all 11 inputs and output.
- [PDF QA receipt](../report/paper_pdf_qa.json) records automated checks.
- [Build/review commands](../README.md#final-submission-audit-and-article-pdf-batch-52)
  retain the separate document-tool environment and unchanged scientific lock.

The next decisions belong to the authors: ethics applicability, journal and
submission revision. No unresolved scientific defect was introduced or hidden by
this author-action follow-up.
