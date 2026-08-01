# AGENTS.md

## What this file is

Codex CLI auto-loads this file at the start of every session in this repo. Nothing else in this project loads automatically — treat this file as your standing memory anchor, and everything else as reachable only if you deliberately go get it.

## Session protocol — do this every session, before anything else

1. **Read `HANDOFF.md`**, top entry (newest first). That tells you what happened last session and what's still open.
2. **Read `CODEX.md`**. That's the current state of project decisions, files, and configs — don't re-derive things that are already decided and logged there.
3. **Read the relevant section(s) of `PROJECT_SPEC.md`** for whatever batch you're being asked to run this session (the user's message will name a batch from `BATCHES.md`; if it doesn't, ask which batch before starting).
4. **Do the work for that batch, and only that batch.** Don't jump ahead to a later phase even if it looks efficient, and don't skip a "stop for review" checkpoint the batch calls for — those checkpoints exist because the decision made there (dataset choice, augmentation handling, training curves) changes what all later phases do.
5. **Before ending the session, or before pausing for review:**
   - Append a new entry to `HANDOFF.md` (template is at the top of that file) — what you did, what's still incomplete, what needs the user's review.
   - Update `CODEX.md` if anything project-level changed: a decision got made, a new config/file was created, a phase finished.
6. **Remind yourself of step 6, always:** `CODEX.md` and `HANDOFF.md` do not load automatically. If you don't deliberately open and update them per steps 1–2 and 5, the next session starts blind and repeats work or re-litigates settled decisions.

## Files in this repo

| File | What it is | Who updates it |
|---|---|---|
| `PROJECT_SPEC.md` | Full requirements spec — research framing, dataset options, hardware-scoped compute budget, phase-by-phase instructions, coding standards, report structure, definition of done. | Nobody — static source of truth. |
| `CODEX.md` | Living project state: decisions made, file map, current phase, open risks. | You, every session. |
| `HANDOFF.md` | Append-only session log. | You, every session. |
| `BATCHES.md` | The sequence of batch instructions the user pastes one at a time. If told "next batch," find your position via `HANDOFF.md` and proceed to the next entry here. | Nobody — static, unless the user asks to revise it. |

## Standing constraints (full detail in `PROJECT_SPEC.md`)

- **Hardware:** RTX 4060 Laptop GPU (8GB VRAM), 16GB system RAM, i7-13650HX. AMP is mandatory. Model scale, batch size, and seed/subsample scoping are fixed in `PROJECT_SPEC.md` §3 — don't renegotiate these mid-project without flagging it to the user first.
- **Exactly two detectors:** Faster R-CNN + one YOLO version. (The source assignment brief says "three two detectors" — that's a typo in the original document, not a real requirement. Build two.)
- **No hardcoding:** class counts, file paths, and hyperparameters all come from `configs/*.yaml`, never inline.
- **Reproducibility:** every number that ends up in the final report must be traceable to a documented command in `README.md`.
- **Known spec quirks** (see `PROJECT_SPEC.md` §0 for full context): the source brief's phase numbering skips a number and the "verify five classes" instruction is a placeholder, not a real constraint — don't hardcode 5 classes for any of the three candidate datasets.

## If something in a batch conflicts with something already logged in `CODEX.md`

Stop and flag it to the user rather than silently picking one. This usually means either the user changed their mind since the last session, or a decision is about to get made twice in two different ways — both are worth a one-line check-in before proceeding.
