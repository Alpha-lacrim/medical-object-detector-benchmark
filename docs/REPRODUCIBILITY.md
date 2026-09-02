# Reproducibility contract

This project distinguishes four different reproducibility claims. Passing a
lower tier does not establish a higher one.

| Tier | What is executed | Required inputs | What success establishes |
|---|---|---|---|
| Software verification | Locked CPU install, Ruff, unit/integration tests, package smoke test, artifact verifier, and manuscript-claim verifier | Git checkout only | The tested software paths work; the committed critical artifacts retain reviewed hashes/schemas; their declared result references exist; and bound manuscript numbers still agree with frozen sources. |
| Committed-analysis reproduction | CPU analysis scripts rerun over committed frozen prediction bundles and provenance summaries | Git checkout and locked CPU environment | Derived tables, figures, and statistics can be recomputed without checkpoint loading, inference, training, or dataset download where the relevant frozen inputs are committed. |
| Exact inference reproduction | Detector inference/evaluation reruns from the exact ten checkpoint hashes and exact processed data/configuration | Licensed RSNA data, processed split, exact external checkpoint files, pinned CUDA environment, and a compatible NVIDIA GPU | Frozen prediction/evaluation outputs can be regenerated from the same model states and inputs. This is not part of standard CI. |
| Exact retraining reproducibility | All ten detector training runs are repeated from their initial pretrained states and recorded seed/configuration | Licensed source data, exact pretrained initialization, pinned CUDA environment, compatible GPU, and the full training budget | The documented training protocol is repeated. It is not a promise of bitwise-identical weights because the recorded CUDA ROI Align backward path is nondeterministic. This is not part of standard CI. |

Green CI therefore verifies software and the integrity/claim consistency of
the committed evidence snapshot. It does not train ten models, download the
RSNA data, rerun GPU inference, or prove that every published scientific result
has been regenerated from raw data.

## Release-candidate verification

Release 2.0.0 uses one version for the Python project/package and the research
release; the immutable release identity is the exact reviewed commit referenced
by tag `v2.0.0`. Before that tag exists, the prepared worktree is a candidate,
not a published release.

The tag also records the then-current text of the tracked
`report/paper_draft.md`, but the manuscript is a living document rather than a
separately versioned or release-frozen publication. It may continue to change on
`main` after v2.0.0. The frozen scientific boundary is the artifact/provenance
inventory verified below; `verify_paper_claims.py` checks the current manuscript
against those sources without making the manuscript itself a frozen artifact.

Run the review gate from the repository root:

```powershell
uv lock --check
uv sync --locked --group dev --extra cpu
uv run --locked --extra cpu ruff format --check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py
uv run --locked --extra cpu ruff check src tests scripts/verify_scientific_artifacts.py scripts/verify_paper_claims.py
uv run --locked --extra cpu python -m pytest -q
uv run --locked --extra cpu python scripts/verify_scientific_artifacts.py
uv run --locked --extra cpu python scripts/verify_paper_claims.py
uv run --locked --extra cpu python -m meddet_benchmark smoke configs/smoke.yaml
git diff --check
```

Foundation CI additionally checks the artifact-manifest builder with Ruff. A
release owner must confirm that the final tag points to the exact commit for
which CI passed. These commands prove the software/integrity boundary in the
table above; they do not promote the run to committed-analysis, exact-inference,
or exact-retraining reproduction.

## Machine-checkable committed evidence

`results/scientific_artifact_manifest.json` is the reviewed inventory of
manuscript-critical artifacts. Each record binds the artifact SHA256 to its
generating script and hash, config and hash, input artifact hashes, frozen
schema, study phase, reproduction tier, and GPU/training requirements. It also
lists result files referenced by the corresponding provenance summary.

The manifest's `canonical_manuscript` field identifies the current manuscript
used by the verification machinery. It does not assign that manuscript a
release version or make its future prose state part of the frozen artifact
inventory.

`report/paper_claim_sources.yaml` binds central numerical manuscript claims to
an exact CSV cell, CSV row count, JSON value, or allow-listed deterministic
calculation and states the permitted absolute rounding tolerance.

Run the same lightweight checks used by CI:

```powershell
& $benchmarkPython scripts/verify_scientific_artifacts.py
& $benchmarkPython scripts/verify_paper_claims.py
```

The artifact verifier fails if a critical artifact, generator, config, or
committed input is missing or stale; if a frozen CSV/JSON/PNG schema changes; or
if a declared result reference is missing. An `external_or_ignored` input may
be absent in a clean checkout, but its hash is checked when it exists. The
claim verifier requires every source to be in the scientific manifest, every
manuscript regex to identify exactly one value, and every comparison to fall
within its stated tolerance.

The manifest is intentionally not regenerated in CI: silently refreshing
hashes would approve changed evidence. After an intentional, reviewed artifact
change, rebuild it locally and inspect the diff:

```powershell
& $benchmarkPython scripts/build_scientific_artifact_manifest.py
git diff -- results/scientific_artifact_manifest.json
```

## Environment and dependency locks

The adopted single-GPU environment is:

- Python 3.11.15 (`.python-version` constrains the project to Python 3.11);
- NumPy 2.4.4 and SciPy 1.17.1;
- Torch 2.6.0+cu124 and Torchvision 0.21.0+cu124;
- Ultralytics 8.4.110;
- the CUDA 12.4 runtime bundled with the Torch wheels and cuDNN 9.1.0; and
- NVIDIA driver 610.47 on an RTX 4060 Laptop GPU with 8,188 MiB reported VRAM.

The machine-wide CUDA 11.2 toolkit is not used by these wheels. The recorded
host has 16 GB RAM and an i7-13650HX.

`pyproject.toml` is the dependency declaration, `uv.lock` is the checked
resolver lock used by CI, `.python-version` fixes the interpreter patch line,
and `requirements.txt` is the exact adopted CUDA-environment package list for
the manual PowerShell setup in `README.md`. CI pins uv 0.11.27, runs
`uv lock --check`, and installs the CPU extra with `uv sync --locked`. GPU
experiments instead install the explicit cu124 Torch/Torchvision wheels before
the remaining exact requirements. Do not interpret a successful CPU lock
install as a test of the CUDA runtime.

Every experiment entry point must call
`initialize_reproducibility(seed, output_dir)` before initializing CUDA. Each
run directory writes `pip_freeze.txt` and `run_environment.json`, including the
seed controls and detected GPU/driver environment. The utility calls
`python -m pip freeze`; if an environment intentionally omits `pip`, it records
that diagnostic and writes an equivalent installed package inventory from
Python package metadata.

## RNG contract and nondeterministic operations

The reproduction initializer:

- records and sets `PYTHONHASHSEED` for child/subsequent interpreters;
- seeds Python, NumPy, Torch CPU, and all CUDA generators;
- sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization;
- disables cuDNN benchmarking and enables deterministic cuDNN behavior; and
- calls `torch.use_deterministic_algorithms(True, warn_only=True)`.

`PYTHONHASHSEED` cannot retroactively change hash randomization in the already
running interpreter. Multi-worker DataLoaders must pass `seed_worker` as
`worker_init_fn` and `make_torch_generator(seed)` as `generator`; each worker's
Python and NumPy state is then derived from the Torch worker seed.

The warning-only policy is deliberate and must not be described as bitwise
determinism. The Batch 2 CUDA smoke run identified Torchvision's
`roi_align_backward_kernel` as lacking a deterministic CUDA implementation in
Torch 2.6.0. It is exercised by Faster R-CNN training and the Grad-CAM path.
Warnings must remain in run logs. Hardware, driver, library, data-loader
scheduling, and floating-point reduction differences can therefore change the
last bits and potentially training trajectories even with fixed RNG state.

On the 16 GB Windows host, Faster R-CNN uses six non-persistent workers for
training and validation. A persistent training pool plus a second six-worker
validation pool exhausted the Windows paging-file/commit limit (`WinError
1455`). Non-persistent workers prevent the pools from coexisting; their startup
time remains inside each epoch duration.

## Exact checkpoint audit and release boundary

`results/checkpoint_release_manifest.json` records the ten validation-selected
best checkpoints consumed by Phase 5. On 2026-08-31 all ten exact local files
were present, all hashes matched the Phase 5 provenance, and their combined
size was 962,924,817 bytes (about 0.90 GiB). The checkpoint binaries remain
Git-ignored and were not committed or uploaded. The manifest's
`public_download_url` is `null`; there is currently no public checkpoint link.

Verify the still-local files against the release manifest without loading
their serialized contents:

```powershell
$release = Get-Content results/checkpoint_release_manifest.json -Raw | ConvertFrom-Json
foreach ($checkpoint in $release.checkpoints) {
    if (-not (Test-Path -LiteralPath $checkpoint.source_path -PathType Leaf)) {
        throw "Missing checkpoint: $($checkpoint.source_path)"
    }
    $actual = (Get-FileHash -LiteralPath $checkpoint.source_path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $checkpoint.sha256) {
        throw "Checkpoint hash mismatch: $($checkpoint.source_path)"
    }
}
```

A public release is assessed as feasible only with the conditions recorded in
the manifest, not as pre-authorized legal clearance. A human release owner
must complete the RSNA attribution/data-policy review, preserve the project's
AGPL obligations for fine-tuned YOLO assets, confirm permission for the
COCO-derived Torchvision initialization used by Faster R-CNN, and inspect
serialized metadata for privacy/security before upload. Do not load untrusted
checkpoint files outside an isolated environment.

If that review succeeds, the release procedure is:

1. Re-run the hash audit above and keep the checkpoint files out of Git.
2. Copy each file to its exact `release_filename` from the manifest.
3. Upload the ten binaries and `checkpoint_release_manifest.json` as external
   release assets under the reviewed license/attribution notices.
4. Download the assets into an isolated staging directory and independently
   verify every size and SHA256.
5. Only after the release exists, add its real immutable URL and release
   identifier to the manifest and documentation. Never insert a speculative
   or placeholder download link.

Until those steps are complete, committed-analysis reproduction remains
available from frozen predictions, while exact checkpoint-based inference
requires access to the audited local files or an independently supplied copy
with matching hashes.
