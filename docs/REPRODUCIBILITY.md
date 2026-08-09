# Reproducibility

Every experiment entry point must call `initialize_reproducibility(seed, output_dir)` before
initializing CUDA. Each run directory then contains `pip_freeze.txt` and
`run_environment.json`, including seed settings and the detected GPU/driver environment.
The utility invokes `python -m pip freeze`; if an environment intentionally omits the `pip`
module, it records that diagnostic and writes the equivalent installed name/version inventory
from Python package metadata instead.

The adopted single-GPU environment is Python 3.11.15, NumPy 2.4.4, SciPy 1.17.1,
Torch 2.6.0+cu124, and Torchvision 0.21.0+cu124. The CUDA 12.4 runtime is bundled
with the Torch wheel and uses NVIDIA driver 610.47; it does not depend on the
machine-wide CUDA 11.2 toolkit. This stack was selected before any detector
training after the originally pinned Python 3.13/CUDA 13.0 wheel download could
not be completed. Both detectors must use this same recorded environment.

Deterministic PyTorch algorithms are enabled by default with warnings for kernels that do not
provide a deterministic implementation. Any such warning must be retained with the run logs and
added here once a concrete model operation triggers it. The Batch 2 CUDA smoke run triggered this
warning for Torchvision's `roi_align_backward_kernel`: it has no deterministic CUDA implementation
in Torch 2.6.0. Training therefore continues under the configured warning-only policy and may have
residual kernel-level nondeterminism despite fixed RNG seeds. `PYTHONHASHSEED` is also recorded, but
it only changes hash randomization for subsequently started Python interpreters.

Multi-worker DataLoaders must pass `seed_worker` as `worker_init_fn` and the result of
`make_torch_generator(seed)` as `generator`. This derives each worker's Python and NumPy seed from
PyTorch's worker seed and keeps later Albumentations calls reproducible.

On this 16 GB Windows host, Faster R-CNN uses six non-persistent workers for
both training and validation. A first official-data benchmark attempt with a
persistent training pool exhausted the Windows paging-file/commit limit when a
second six-worker validation pool imported PyTorch (`WinError 1455`). Disabling
persistence prevents the two pools from coexisting; worker startup is retained
inside every epoch's measured duration.
