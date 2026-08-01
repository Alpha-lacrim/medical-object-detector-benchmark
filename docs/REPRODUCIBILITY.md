# Reproducibility

Every experiment entry point must call `initialize_reproducibility(seed, output_dir)` before
initializing CUDA. Each run directory then contains `pip_freeze.txt` and
`run_environment.json`, including seed settings and the detected GPU/driver environment.
The utility invokes `python -m pip freeze`; if an environment intentionally omits the `pip`
module, it records that diagnostic and writes the equivalent installed name/version inventory
from Python package metadata instead.

Deterministic PyTorch algorithms are enabled by default with warnings for kernels that do not
provide a deterministic implementation. Any such warning must be retained with the run logs and
added here once a concrete model operation triggers it. `PYTHONHASHSEED` is also recorded, but it
only changes hash randomization for subsequently started Python interpreters.

Multi-worker DataLoaders must pass `seed_worker` as `worker_init_fn` and the result of
`make_torch_generator(seed)` as `generator`. This derives each worker's Python and NumPy seed from
PyTorch's worker seed and keeps later Albumentations calls reproducible.
