# Experiment records

This directory is maintained by `tools/run_experiment.py` and
`tools/finalize_experiment.py`.

- `runs/<run_id>/` contains immutable per-run evidence and manifests.
- `runs.csv` indexes all finalized runs by `run_id`.
- `evidence_manifest.tsv` indexes artifact hashes.
- `tables/*.csv` files are the machine-readable sources of truth.
- `tables/*.md` files are generated from the corresponding CSV files.
- `tables/pcc_ablation.csv` records explicit cross-camera-positive and PCC
  weights separately while retaining fixed-index-compatible fields.
- `tables/alignment_ablation.csv` compares formal fixed-index and hard-alignment
  runs. Its Markdown view is always generated from the CSV source.

Registry schema version 2 adds `run_kind`, method/alignment identity fields,
and hard-alignment evidence. Existing version-1 CSV/TSV rows are migrated
losslessly: historical fields and rows are retained, legacy runs default to
`run_kind=formal`, and unavailable evidence remains explicitly marked.

Failed or incomplete runs remain under `runs/`, but they are not written to
successful result tables. Successful smoke runs are indexed in `runs.csv` and
`evidence_manifest.tsv`, while all formal result tables and the generated
section of `EXPERIMENTS.md` exclude them. Historical rows outside that section
are never rewritten.

Formal training applies the shared protocol in `utils/reproducibility.py` before
constructing DataLoaders, models, or optimizers, then writes the actual receipt
to `OUTPUT_DIR/reproducibility.json`. The recorder remains fail-closed: it only
copies and validates that training-produced evidence and never infers a seed.

The runner requires a completely clean worktree before initialization. After it
creates `runs/<run_id>/`, Git verification permits only new files beneath that
exact run directory until finalization; tracked changes or any other untracked
path still fail closed.
