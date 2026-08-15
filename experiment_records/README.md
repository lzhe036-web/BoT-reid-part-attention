# Unified Experiment Records

This directory uses the single project schema v5 while retaining historical
v1-v4 evidence under its original schema identity. The migration reader
preserves every old row and unknown old column; evidence that cannot be
reconstructed is represented as `not_recorded`, never invented or relabelled v5.

The authoritative sources for generated Markdown are `runs.csv`,
`tables/main_results.csv`, `evidence_manifest.tsv`, and each run's
`run_manifest.json`, `run_status.json`, and `checkpoint_manifest.tsv`.

- **Run Registry / All Recorded Runs** contains every initialized, running,
  training-complete, finalizing, successful, failed, incomplete, and interrupted run.
- **Formal Results** contains only `run_kind=formal` and `status=success`.
- **Checkpoint Evidence** contains one row for every real checkpoint, using
  Ignite `EPOCH_EVIDENCE` for epoch/global-iteration binding.

`EXPERIMENTS.md` is the only project Markdown registry. Its sole generated
marker set is `AUTO-EXPERIMENT-RUNS`, `AUTO-EXPERIMENT-RESULTS`, and
`AUTO-CHECKPOINT-EVIDENCE`. Sections are replaced atomically and idempotently
while all manually maintained text is preserved. Legacy Dynamic-only markers
may be removed only after every contained run_id is present in the authoritative
v5 tables. No lowercase or alternate-casing experiment Markdown file is created.

Dynamic Gating scalar statistics are written directly to `runs.csv`, formal
results (when eligible), and generated Markdown. Bounded per-sample evidence is
kept in `dynamic_gating_summary.json` and `gating_samples.tsv`; their path,
size, SHA256, source-checkpoint SHA256, and deterministic selection rule are
recorded in every registry and manifest.

Formal execution is unlocked only by a schema-v5 successful one-epoch smoke
whose control files and every artifact hash validate, and whose normalized
protocol, implementation, parent/shared-feature, gating, Seed42 and K=[2,4,6]
signatures match the current candidate. A later commit is accepted only when
its Git diff contains recorder-managed evidence products exclusively.

`run_manifest.json` and `run_status.json` avoid self-reference: after both are
written, the registry hashes them and records their path, size and SHA256 in
`runs.csv`, the global evidence manifest and `EXPERIMENTS.md`. The artifact
manifest explicitly identifies these two files as externally sealed controls.

Missing values have strict meanings:

- `not_applicable`: the field does not apply.
- `not_recorded`: the artifact or metric has not yet been produced.
- `missing_evidence`: it should exist for the current state but is absent.
