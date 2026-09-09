#!/usr/bin/env python
"""Check one-epoch smoke identity before launching a formal alpha experiment."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from config import cfg
from utils.config_serialization import deserialize_cfg_node_yaml
from utils.g2_e_checkpoint_identity import validate
from utils.g2_e_alpha_delivery import checked_file, read_json, read_rows
from utils.dynamic_experiment_registry import sha256_file


def verify(alpha, config_file, smoke_output):
    token={.1:'0p1',.5:'0p5'}[alpha]
    root=Path(smoke_output).resolve()
    source=cfg.clone();source.merge_from_file(str(config_file))
    if (source.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_ALPHA!=alpha
            or source.MODEL.MULTI_GRANULARITY_GATING_TAU!=.5
            or not source.MODEL.MULTI_GRANULARITY_STATIC_DYNAMIC_RESIDUAL):
        raise ValueError('Smoke source alpha/tau/fusion mismatch')
    expected=source.clone()
    expected.SOLVER.MAX_EPOCHS=1;expected.SOLVER.CHECKPOINT_PERIOD=1;expected.SOLVER.EVAL_PERIOD=1
    expected.OUTPUT_DIR=str(root)
    resolved_path=root/'config_resolved.yml'
    resolved=deserialize_cfg_node_yaml(resolved_path.read_text(encoding='utf-8'))
    if expected!=resolved:
        raise ValueError('Smoke resolved configuration differs beyond declared one-epoch/output overrides')
    provenance=read_json(root/'reproducibility.json')
    current=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if (provenance['code']['commit']!=current or provenance['code']['dirty']
            or provenance['code']['branch']!='codex/g2-e-static-dynamic-alpha{}-tau0p5'.format(token)
            or provenance['configuration']['source_file_sha256']!=sha256_file(config_file)
            or provenance['configuration']['resolved_file_sha256']!=sha256_file(resolved_path)):
        raise ValueError('Smoke training/configuration identity is stale or inconsistent')
    manifest=read_json(root/'g2_e_smoke_analysis'/('g2_e_static_dynamic_alpha{}_tau0p5_analysis_manifest.json'.format(token)))
    metadata=validate(manifest['checkpoint_path'],source)
    if metadata['epoch']!=1 or manifest['checkpoint_sha256']!=metadata['checkpoint_sha256']:
        raise ValueError('Smoke checkpoint/epoch mismatch')
    for evidence in manifest['files'].values():checked_file(evidence)
    rows=read_rows(manifest['files']['per_sample_gating_tsv']['path'],'\t')
    if len(rows)!=32 or any(float(r['alpha'])!=alpha or r['checkpoint_sha256']!=metadata['checkpoint_sha256'] for r in rows):
        raise ValueError('Expected 32 smoke samples from this alpha checkpoint')
    return {'status':'verified','kind':'smoke_only','alpha':alpha,'training_commit':current}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha',required=True,type=float,choices=(.1,.5))
    parser.add_argument('--config-file',required=True)
    parser.add_argument('--smoke-output',required=True)
    print(json.dumps(verify(**vars(parser.parse_args(argv))),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
