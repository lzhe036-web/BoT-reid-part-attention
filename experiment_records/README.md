# Experiment records

This directory is maintained by `tools/run_experiment.py` and
`tools/finalize_experiment.py`.

- `runs/<run_id>/` contains immutable per-run evidence and manifests.
- `runs.csv` indexes all finalized runs by `run_id`.
- `evidence_manifest.tsv` indexes artifact hashes.
- `tables/*.csv` files are the machine-readable sources of truth.
- `tables/*.md` files are generated from the corresponding CSV files.
- `tables/pcc_ablation.csv` records explicit cross-camera-positive and PCC
  weights separately for fixed-index PCC runs.
- `tables/granularity_fusion_ablation.csv` compares the static fusion control
  with per-sample dynamic gating and links trained-checkpoint gate evidence.

Failed or incomplete runs remain under `runs/`, but they are not written to
successful result tables. Historical rows outside the generated section of
`EXPERIMENTS.md` are never rewritten.

Formal training applies the shared protocol in `utils/reproducibility.py` before
constructing DataLoaders, models, or optimizers, then writes the actual receipt
to `OUTPUT_DIR/reproducibility.json`. The recorder remains fail-closed: it only
copies and validates that training-produced evidence and never infers a seed.

The runner requires a completely clean worktree before initialization. After it
creates `runs/<run_id>/`, Git verification permits only new files beneath that
exact run directory until finalization; tracked changes or any other untracked
path still fail closed.
