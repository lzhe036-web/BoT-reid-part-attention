# C2-L03 Multi-Granularity Part Experiment Records

This directory is the only repository-level registry for experiment family
`C2-MGP-K246`. It is independent of every legacy experiment table.

Formal execution uses:

```bash
bash scripts/train_c2_l03_multi_granularity_part_autodl.sh
```

The unified runner performs a clean-repository preflight, locks every formal
protocol field, records environment, seed and independent DataLoader generator
streams, dataset/model manifests, and mandatory CUDA efficiency metadata. It
then executes training, measures each stage with `time.monotonic`, selects one structured
validation block, binds it to a model checkpoint, verifies hashes, and only
then updates `runs.csv` and `evidence_manifest.tsv`.

`run_id` is the idempotency key: an identical repeated finalization is a no-op;
changed content or hashes are rejected. Registry candidates are staged first,
and `runs.csv` is committed last as the success marker. Failed or
incomplete runs remain only in their own `OUTPUT_DIR` and never enter the
success registry.

`training_runtime_seconds` measures only the `tools/train.py` subprocess and is
the runtime used for method comparison. `total_run_runtime_seconds` includes
the surrounding evidence workflow. A locally complete run remains
`local_complete_pending_commit_and_archive` until the result commit and an
external archive are recorded.

Runtime fields use the following boundaries:

- `environment_collection_runtime_seconds`: environment package/hardware query.
- `profiling_runtime_seconds`: mandatory profiler subprocess only.
- `training_runtime_seconds`: `tools/train.py` subprocess only.
- `finalization_runtime_seconds`: evidence validation through metrics and
  checkpoint generation, draft artifact hashing, and registry candidate
  staging. Its explicit cutoff is immediately before the second-pass final
  hashing and atomic registry seal, which are excluded to avoid a
  runtime/manifest/hash self-reference.
- `total_run_runtime_seconds`: one monotonic interval beginning at `_run_formal`
  before config loading/preflight and ending at the same explicit finalization
  boundary above; the final second-pass atomic seal is excluded and is not
  represented as measured runtime.

Missing-value meanings:

- `not_recorded`: expected evidence has not been measured or committed.
- `not_archived`: the evidence exists locally but no external archive is recorded.
- `not_applicable`: the field does not apply to this run.

Dry-run fixture validation never starts training or accesses a dataset:

```bash
python tools/run_c2_l03_multi_granularity_part.py \
  --dry-run --fixture-dir /path/inside/system/temp/fixture
```

The fixture must contain only synthetic artifacts following `schema.json`.
