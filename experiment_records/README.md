# Unified Experiment Records

This directory contains schema-v2 experiment evidence while retaining the
historical schema-v1 registry under `c2_l03_multi_granularity_part/` unchanged.
The migration reader preserves every old row and unknown old column; evidence
that cannot be reconstructed is represented as `not_recorded`, never invented.

The authoritative sources for generated Markdown are `runs.csv`,
`tables/main_results.csv`, `evidence_manifest.tsv`, and each run's
`run_manifest.json`, `run_status.json`, and `checkpoint_manifest.tsv`.

- **Run Registry / All Recorded Runs** contains every initialized, running,
  training-complete, successful, failed, incomplete, and interrupted run.
- **Formal Results** contains only `run_kind=formal` and `status=success`.
- **Checkpoint Evidence** contains one row for every real checkpoint, using
  Ignite `EPOCH_EVIDENCE` for epoch/global-iteration binding.

`EXPERIMENTS.md` is the only project Markdown registry. Generated marker
sections are replaced atomically and idempotently while all manually maintained
text is preserved. No lowercase or alternate-casing experiment Markdown file is
created.

Dynamic Gating scalar statistics are written directly to `runs.csv`, formal
results (when eligible), and generated Markdown. Bounded per-sample evidence is
kept in `dynamic_gating_summary.json` and `gating_samples.tsv`; their path,
size, SHA256, source-checkpoint SHA256, and deterministic selection rule are
recorded in every registry and manifest.

Missing values have strict meanings:

- `not_applicable`: the field does not apply.
- `not_recorded`: the artifact or metric has not yet been produced.
- `missing_evidence`: it should exist for the current state but is absent.
