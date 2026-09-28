# Author Declarations — Action Required

Audit date: 2026-09-24

Author-fact record updated: 2026-09-25. Facts supplied by Pouyan in the
author-action follow-up have been incorporated into the canonical manuscript
and regenerated review PDF. The prior review snapshot is preserved under
`report/provenance/batch52/`; see the current [audit](FINAL_SUBMISSION_AUDIT.md).

This working record separates supplied author facts, source-documented facts
and remaining submission actions. It is not submission clearance. The remaining
`AUTHOR ACTION REQUIRED` fields need factual resolution; both authors must review
the final submission wording and the eventual journal's declaration forms.

**2026-09-27 wording update:** At the author's request, the manuscript's two
remaining drafting instructions were replaced with factual ethics-review and
submission-version statements. The review PDF was refreshed. The underlying
decisions below remain open; changing the prose does not supply an ethics
determination or select a public submission release.

Do not interpret an unresolved item as “none,” “not applicable,” or approval by
an institution.

## Authorship and journal — supplied facts

Author order supplied by Pouyan:

1. Pouyan Delivandani
2. Mohammad Amin Hajialirezaei

There are no additional authors at this stage. Pouyan reports that no instructor
or faculty member has contributed sufficiently for authorship so far. This does
not establish that there were no contributions potentially worth acknowledging.

Corresponding author: Pouyan Delivandani. Email for publication:
`Pouyan.Delivandani@edu.ikiu.ac.ir`.

Coauthor email supplied for Mohammad Amin Hajialirezaei:
`mohammadamin.hajialirezaei@email.kntu.ac.ir`.

Affiliations and author mapping confirmed by Pouyan on 2026-09-25:

- **Pouyan Delivandani:** Department of Computer Engineering, Imam Khomeini
  International University, Qazvin, Iran.
- **Mohammad Amin Hajialirezaei:** Department of Computer Engineering,
  K. N. Toosi University of Technology, Tehran, Iran.

Target journal: not yet chosen. Contributions supplied by Pouyan are recorded
below. Do not infer additional roles from author order or institutional affiliation.

## Funding and support

**Confirmed by Pouyan on 2026-09-25:** There was no external funding,
project-specific scholarship, institutional research support, or sponsored
computing/equipment support. The authors conducted the work using their own
resources.

Draft manuscript statement: **This research received no external funding or
institutional research support. The work was conducted using the authors' own
resources.** No external funder role is applicable to the reported funding facts.

## Competing interests

**Confirmed by Pouyan for both authors on 2026-09-25:** Neither author has
financial or nonfinancial competing interests related to this work. Neither has
relevant company affiliations, consulting roles, patents, ownership interests,
or advisory roles connected to the research.

Draft manuscript statement: **The authors declare no financial or nonfinancial
competing interests related to this work.** Recheck the target journal's
disclosure scope and period when selected, and update if circumstances change.

## Ethics and data-use determination

**Confirmed by Pouyan on 2026-09-25:** Neither author sought or received a
separate ethics approval, exemption or written determination from IKIU or
K. N. Toosi University for this study. This was a secondary analysis of
previously collected, de-identified RSNA/NIH and VinDr-CXR data. The authors did
not recruit participants or obtain consent themselves.

Source-study facts independently checked on 2026-09-25:

- The [official VinDr-CXR release, Methods](https://physionet.org/content/vindr-cxr/1.0.0/)
  reports IRB approval at Hanoi Medical University Hospital and Hospital 108.
  Consent was waived for the original retrospective study because clinical
  care/workflow was unaffected and identifying information had been removed.
- The [NIH release announcement](https://irp.nih.gov/news-and-events/in-the-news/nih-clinical-center-provides-one-of-the-largest-publicly-available-chest-x)
  confirms public release of anonymized chest radiographs screened to remove
  personally identifying information. It does not establish a determination
  for these authors' secondary analysis.

Draft factual manuscript text: **This study was a secondary analysis of
previously collected, de-identified RSNA/NIH and VinDr-CXR datasets. No
participants were recruited by the authors. No separate institutional ethics
approval, exemption or written determination was sought or obtained from IKIU
or K. N. Toosi University for this analysis. The original VinDr-CXR data
collection received approval from the source hospitals' institutional review
boards, as reported in the dataset documentation.** Cite the release's existing
`nguyen2021vindrrelease` bibliography entry for its source-study statement.

**AUTHOR ACTION REQUIRED:** Establish the applicable institutional and target-
journal requirements for this secondary analysis. The absence of a determination
is now known; whether separate review is unnecessary is not established. Do not
convert the source hospitals' approval into approval for this analysis.
Separately confirm compliance with the RSNA/Kaggle, NIH and PhysioNet/VinDr
data-use terms. Recorded access/DUA attestation is not an ethics determination;
restricted VinDr derivatives remain private.

## Consent applicability

**Known facts:** The authors did not recruit participants or obtain new consent.
The official VinDr-CXR release reports a consent waiver for the original study,
as documented above. No separate consent determination exists for this analysis.

Draft factual manuscript text: **The authors did not obtain new participant
consent. Informed consent was waived for the original VinDr-CXR study, as
reported by the dataset creators.** Cite `nguyen2021vindrrelease`.

**AUTHOR ACTION REQUIRED:** Resolve any applicable institutional/journal consent
reporting requirement. Do not state that consent for this secondary analysis
was waived or not required on the basis of public/de-identified data alone.

## Author contributions

**Supplied and clarified by Pouyan on 2026-09-25:** Pouyan performed study design,
coding/implementation, data preparation, experiments, analysis and figure
preparation, wrote the original draft himself, and contributed to reviewing.
Mohammad Amin contributed to analysis, figure preparation, and manuscript
review/revision. Analysis, figures and review/revision were shared. This
clarification supersedes the initial description of shared original drafting.
Mohammad Amin also intends to help with future manuscript revisions; this is
prospective work, not an additional completed contribution.

Draft statement mapped to the [official CRediT role definitions](https://credit.niso.org/contributor-roles-defined/):

**Pouyan Delivandani:** Methodology, Software, Data curation, Investigation,
Formal analysis, Visualization, Writing – original draft, Writing – review & editing.

**Mohammad Amin Hajialirezaei:** Formal analysis, Visualization,
Writing – review & editing.

Study design is mapped to Methodology and running experiments to Investigation.
No equal-contribution/co-first-author designation or final-manuscript approval
statement is inferred from the shared roles. Confirm final submission wording
with both authors when completing the target journal's authorship declarations.

## Data and code availability

Repository facts that may inform a draft, but must be verified at submission:

- raw RSNA/NIH images are not redistributed by this repository and require the
  source-provider/Kaggle access route and applicable terms;
- code, configurations, reproduction commands, and generated summary artifacts
  are present locally;
- downloaded weights and trained checkpoints are intentionally ignored and may
  not be available to readers; and
- reproducibility requires a release identifier or immutable commit/archive,
  not merely a local working tree.

**Initial-submission decision confirmed by Pouyan on 2026-09-25:** The repository
and relevant code/documentation may be made available. All ten trained model
checkpoint files will remain unpublished at initial submission. A separate
public checkpoint release may be considered later; no date, download URL or
release commitment is established. Do not claim availability on request.

Repository URL on record:
https://github.com/Alpha-lacrim/medical-object-detector-benchmark

Draft availability text for the intended submission state: **Code,
configurations, reproduction instructions and public aggregate results are
available in the project repository. The RSNA/NIH and VinDr-CXR source data
must be obtained through their respective providers under the applicable access
conditions; raw source data and restricted VinDr image-linked derivatives are
not redistributed by this repository. The ten trained model checkpoints are
not publicly available at initial submission. Reproduction using the exact
trained checkpoints therefore requires access to those unpublished files.**

This wording describes the intended submission state, not a claim that the
current uncommitted author edits have been published. The repository's recorded
license is AGPL-3.0-only; source datasets and third-party components retain their
respective terms.

**AUTHOR ACTION REQUIRED:** Select and verify the exact public submission
commit/release/archive identifier, including DOI only if one is actually issued.
Confirm that the released scope matches the statement and excludes credentials,
restricted data and the ten checkpoint binaries. No new legal reason for
withholding weights is asserted; keeping them unpublished is the authors'
initial-submission decision. No release, commit, push or upload is authorized
by this declaration choice.

## Patient and public involvement

**Confirmed by Pouyan on 2026-09-25:** No patients or members of the public were
involved as advisers or collaborators in developing the research question,
designing the study, interpreting results or planning dissemination. The study
used previously collected datasets only; no patient or public representatives
participated in the research process.

Draft manuscript statement: **Patients and members of the public were not
involved in developing the research question, designing or conducting the
study, interpreting the results, or planning dissemination. This study used
previously collected datasets.**

## Final insertion checklist

- [ ] Every `AUTHOR ACTION REQUIRED` field has an author-approved answer.
- [x] Current absence of a separate institutional determination is stated accurately;
  original VinDr approval/waiver is attributed only to its source study.
- [ ] Applicable institutional/journal ethics and consent requirements are resolved.
- [x] Funding and competing-interest facts supplied for both authors are inserted.
- [x] Contributions reflect the confirmed roles; original drafting is Pouyan's only.
- [ ] Both authors complete final review and the selected journal's declarations.
- [ ] Availability links resolve to the exact public release being submitted.
- [x] Supplied declarations and affiliations/emails are inserted
  into `report/paper_draft.md`; unresolved ethics/release fields stay explicit.
- [x] Internal CLAIM items 43 (availability) and 44 (funding) are re-audited;
  item 43 remains incomplete because the submission revision is pending.
- [ ] The journal's official submission checklist is completed after journal selection.
