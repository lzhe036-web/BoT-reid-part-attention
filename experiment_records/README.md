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

Failed or incomplete runs remain under `runs/`, but they are not written to
successful result tables. Historical rows outside the generated section of
`EXPERIMENTS.md` are never rewritten.

The current legacy trainer does not expose proof of an applied training seed.
Because the recorder is strictly read-only, it records that fact as
`missing_evidence` and refuses to publish a successful formal row; it does not
seed the trainer, infer a seed, or reuse a historical result.
