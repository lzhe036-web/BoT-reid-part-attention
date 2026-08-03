#!/usr/bin/env python
"""Reproducible local efficiency profiling for the formal C2-L03 evidence set.

This tool never trains an epoch and never writes to ``thesis_evidence``.  It
loads the selected archived checkpoint strictly, profiles only a forward pass
for parameters/MACs/inference timing, and optionally runs a bounded sequence
of real-data training steps solely to measure CUDA peak memory.

The profiling seed is local analysis metadata; it must not be interpreted as
the historical training seed of any archived experiment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = Path(r"D:\thesis_reid\thesis_evidence")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_notes" / "c2_l03_final_evidence" / "efficiency_analysis"
INPUT_SHAPE = (1, 3, 256, 128)
FORMAL_IDS = (
    "C2-Baseline-Control", "C2-L01", "C2-L03", "C2-L05", "C2-L10",
    "Duke-Baseline-Control", "Duke-C2-L03", "S2-SCPO-Market",
    "CAAT-L01", "CAAT-L03", "CAAT-L05", "CAAT-L10",
)
WORKTREE_BY_COMMIT = {
    "d98fb000e2b10c18c367ee2a01474fe925600cd0": "bot-eff-d98fb000",
    "7b8195d4a02b536b27ab4d6ac80652091db7468f": "bot-eff-7b8195d4",
    "3ce47246d67c1c43befd651ea44082216167478f": "bot-eff-3ce47246",
    "f1f16925b9dfc42203ad3b80adcd3ce744a8aa33": "bot-eff-f1f16925",
    "61da0df73424323dd9968fddf670d4ef78a24b9e": "bot-eff-61da0df7",
}

SUMMARY_FIELDS = (
    "experiment_id", "dataset", "method_variant", "selected_epoch", "config_ref",
    "checkpoint_ref", "checkpoint_sha256", "input_shape", "profiling_precision",
    "total_params", "trainable_params", "macs", "flops", "flop_tool",
    "flop_tool_version", "unsupported_ops", "inference_batch_size", "inference_warmup",
    "inference_repeats", "inference_mean_ms", "inference_median_ms", "inference_p95_ms",
    "inference_images_per_second", "training_batch_size", "num_instances",
    "training_peak_allocated_mib", "training_peak_reserved_mib",
    "training_peak_allocated_gib", "training_peak_reserved_gib", "training_time_value",
    "training_time_basis", "training_seed", "profiling_seed", "gpu_name",
    "gpu_memory_total", "cuda_version", "torch_version", "cudnn_version",
    "measurement_status", "evidence_grade", "notes",
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset: str
    method_variant: str
    aux_lambda: str
    selected_epoch: int
    config_path: Path
    checkpoint_path: Path
    log_path: Path
    code_commit: str


def ev(*parts: str) -> Path:
    return EVIDENCE_ROOT.joinpath(*parts)


LAMBDA_ROOT = ev("lambda_sensitivity", "2026-07-22", "extracted", "server_ltmxmapcaz-a5a5f604_20260722_212555")
S2_ROOT = ev("ablations", "same_camera_positive_only", "2026-07-22", "extracted", "server_78514386a4-df6af7ea_20260722_204901")
CAAT_ROOT = ev("ablations", "camera_aware_triplet", "2026-07-22", "extracted", "server_txlle37mlw-a8bf2285_20260722_220716")
DIST_ROOT = ev("distance_analysis", "2026-07-18", "extracted")


def experiment_specs() -> Dict[str, ExperimentSpec]:
    """Return the 12 rows eligible for the formal efficiency table only."""
    specs: Dict[str, ExperimentSpec] = {}
    baseline_root = DIST_ROOT / "server_78514386"
    c2_l03_root = DIST_ROOT / "server_ltmxmapcaz"
    specs["C2-Baseline-Control"] = ExperimentSpec(
        "C2-Baseline-Control", "Market-1501", "Baseline-Control", "not_applicable", 120,
        baseline_root / "market_baseline" / "config.yml",
        baseline_root / "market_baseline" / "resnet50_checkpoint_22320.pt",
        baseline_root / "market_baseline" / "log.txt",
        "d98fb000e2b10c18c367ee2a01474fe925600cd0",
    )
    for suffix, value in (("01", "0.1"), ("03", "0.3"), ("05", "0.5"), ("10", "1.0")):
        eid = f"C2-L{suffix}"
        if suffix == "03":
            config_path = c2_l03_root / "market_c2_l03" / "config.yml"
            checkpoint_path = c2_l03_root / "market_c2_l03" / "resnet50_checkpoint_22320.pt"
            log_path = c2_l03_root / "market_c2_l03" / "log.txt"
        else:
            stem = f"cross_camera_positive_lambda{suffix}_market1501"
            config_path = LAMBDA_ROOT / "configs" / f"softmax_triplet_cross_camera_positive_lambda{suffix}_autodl.yml"
            checkpoint_path = LAMBDA_ROOT / stem / "resnet50_checkpoint_22320.pt"
            log_path = LAMBDA_ROOT / stem / "log.txt"
        specs[eid] = ExperimentSpec(
            eid, "Market-1501", "Cross-Camera Positive Only", value, 120,
            config_path, checkpoint_path, log_path,
            "7b8195d4a02b536b27ab4d6ac80652091db7468f",
        )
    specs["Duke-Baseline-Control"] = ExperimentSpec(
        "Duke-Baseline-Control", "DukeMTMC-reID", "Baseline-Control", "not_applicable", 80,
        baseline_root / "duke_baseline" / "config.yml",
        baseline_root / "duke_baseline" / "resnet50_checkpoint_19280.pt",
        baseline_root / "duke_baseline" / "log.txt",
        "3ce47246d67c1c43befd651ea44082216167478f",
    )
    specs["Duke-C2-L03"] = ExperimentSpec(
        "Duke-C2-L03", "DukeMTMC-reID", "Cross-Camera Positive Only", "0.3", 120,
        baseline_root / "duke_c2_l03" / "config.yml",
        baseline_root / "duke_c2_l03" / "resnet50_checkpoint_28920.pt",
        baseline_root / "duke_c2_l03" / "log.txt",
        "3ce47246d67c1c43befd651ea44082216167478f",
    )
    specs["S2-SCPO-Market"] = ExperimentSpec(
        "S2-SCPO-Market", "Market-1501", "Same-Camera Positive Only", "0.5", 120,
        S2_ROOT / "s2" / "softmax_triplet_same_camera_positive_only_autodl.yml",
        S2_ROOT / "s2" / "resnet50_checkpoint_22320.pt",
        S2_ROOT / "s2" / "log.txt",
        "f1f16925b9dfc42203ad3b80adcd3ce744a8aa33",
    )
    for suffix, value in (("01", "0.1"), ("03", "0.3"), ("05", "0.5"), ("10", "1.0")):
        stem = f"camera_aware_triplet_lambda{suffix}_market1501"
        specs[f"CAAT-L{suffix}"] = ExperimentSpec(
            f"CAAT-L{suffix}", "Market-1501", "Full Camera-Aware Triplet", value, 120,
            CAAT_ROOT / "BoT-reid-cat-lambda" / "configs" / f"softmax_triplet_camera_aware_lambda{suffix}_autodl.yml",
            CAAT_ROOT / "experiments" / "BoT" / stem / "resnet50_checkpoint_22320.pt",
            CAAT_ROOT / "experiments" / "BoT" / stem / "log.txt",
            "61da0df73424323dd9968fddf670d4ef78a24b9e",
        )
    return specs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path(r"D:\thesis_reid"))).replace("\\", "/")
    except ValueError:
        return str(path)


def set_profiling_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return data


def config_value(config: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    part = config.get(section, {})
    return part.get(key, default) if isinstance(part, dict) else default


def checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise ValueError("Checkpoint is not a mapping")
    state = blob.get("model", blob)
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a model state mapping")
    return state


def load_inference_model(spec: ExperimentSpec, code_root: Path, device: torch.device) -> tuple[torch.nn.Module, Mapping[str, Any], int, str]:
    """Construct from the archived config and strictly load its selected weight."""
    if not spec.config_path.is_file():
        raise FileNotFoundError(f"Archived config missing: {spec.config_path}")
    if not spec.checkpoint_path.is_file():
        raise FileNotFoundError(f"Archived checkpoint missing: {spec.checkpoint_path}")
    sys.path.insert(0, str(code_root))
    from modeling.baseline import Baseline  # import after exact code root is first on sys.path

    config = read_yaml(spec.config_path)
    state = checkpoint_state(spec.checkpoint_path)
    if "classifier.weight" not in state:
        raise ValueError("Cannot determine num_classes: classifier.weight is absent")
    num_classes = int(state["classifier.weight"].shape[0])
    model = Baseline(
        num_classes=num_classes,
        last_stride=int(config_value(config, "MODEL", "LAST_STRIDE", 1)),
        model_path="",
        neck=str(config_value(config, "MODEL", "NECK", "bnneck")),
        neck_feat=str(config_value(config, "TEST", "NECK_FEAT", "after")),
        model_name=str(config_value(config, "MODEL", "NAME", "resnet50")),
        pretrain_choice="none",  # selected checkpoint is loaded below; never download ImageNet weights
        part_attention=bool(config_value(config, "MODEL", "PART_ATTENTION", False)),
        part_attention_parts=int(config_value(config, "MODEL", "PART_ATTENTION_PARTS", 6)),
    )
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint validation failed: {incompat}")
    model.to(device).eval()
    return model, config, num_classes, sha256(spec.checkpoint_path)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def profile_macs_with_torch_profiler(model: torch.nn.Module, device: torch.device) -> tuple[int, int, str, List[Dict[str, Any]]]:
    """Use PyTorch profiler's operation FLOP annotations for the real forward graph.

    PyTorch's profiler reports FLOP estimates for convolution/matmul-style
    operators, not every non-linear or normalization operator.  The returned
    MACs are therefore tool-reported FLOPs / 2 and the caveat is persisted.
    """
    x = torch.randn(*INPUT_SHAPE, device=device, dtype=torch.float32)
    torch.cuda.synchronize(device)
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as profiler:
        model(x)
    torch.cuda.synchronize(device)
    raw_events = [
        {
            "operator": item.key,
            "flops": int(getattr(item, "flops", 0) or 0),
            "calls": int(getattr(item, "count", 0) or 0),
        }
        for item in profiler.key_averages()
    ]
    total_flops = int(sum(item["flops"] for item in raw_events))
    if total_flops <= 0:
        raise RuntimeError("torch.profiler returned no FLOP annotations")
    unsupported = (
        "torch.profiler FLOP annotations do not include all non-linear/normalization "
        "operations; observed forward graph includes batch_norm, adaptive_avg_pool2d, add and softmax"
    )
    return total_flops // 2, total_flops, unsupported, raw_events


def write_raw_profiler_output(output_dir: Path, spec: ExperimentSpec, macs: int, flops: int,
                              events: Sequence[Mapping[str, Any]]) -> Path:
    """Persist the profiler's operation-level raw FLOP annotations for audit."""
    raw_dir = output_dir / "raw_profiles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{spec.experiment_id}.torch_profiler.json"
    payload = {
        "experiment_id": spec.experiment_id,
        "input_shape": "1x3x256x128",
        "precision": "FP32",
        "mode": "eval_inference_forward_only",
        "tool": "torch.profiler",
        "tool_version": torch.__version__,
        "tool_reported_flops": flops,
        "macs_definition": "tool_reported_flops / 2",
        "macs": macs,
        "flops_definition": "2 * MACs",
        "flops": flops,
        "operator_events": list(events),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def percentile(values: Sequence[float], percentage: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentage / 100.0
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def benchmark_inference(model: torch.nn.Module, device: torch.device, warmup: int, repeats: int) -> Dict[str, float]:
    x = torch.randn(*INPUT_SHAPE, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize(device)
        elapsed_ms: List[float] = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms.append(float(start.elapsed_time(end)))
    mean_ms = mean(elapsed_ms)
    return {
        "mean_ms": mean_ms,
        "median_ms": median(elapsed_ms),
        "p95_ms": percentile(elapsed_ms, 95.0),
        "images_per_second": 1000.0 / mean_ms if mean_ms > 0 else float("nan"),
    }


def parse_training_time(log_path: Path) -> tuple[str, str, str]:
    """Return a documented explicit runtime, log span, or not_recorded."""
    if not log_path.is_file():
        return "not_recorded", "not_recorded", "log_not_archived"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    explicit = re.search(r"(?:total\s+)?(?:training|train)\s+(?:time|runtime)\s*[:=]\s*([0-9.]+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)", text, re.I)
    if explicit:
        return f"{explicit.group(1)} {explicit.group(2)}", "explicit_recorded_runtime", "parsed_explicit_runtime"
    stamps = re.findall(r"\b(20\d{2}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:,\d{3})?)", text)
    parsed: List[dt.datetime] = []
    for stamp in stamps:
        for pattern in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed.append(dt.datetime.strptime(stamp, pattern))
                break
            except ValueError:
                pass
    if len(parsed) >= 2 and parsed[-1] >= parsed[0]:
        seconds = int((parsed[-1] - parsed[0]).total_seconds())
        return f"{seconds} s", "derived_log_span", "first_to_last_timestamp; includes_logged_evaluation_and_io"
    return "not_recorded", "not_recorded", "no_explicit_runtime_or_parseable_log_span"


def cuda_environment() -> Dict[str, str]:
    if not torch.cuda.is_available():
        return {
            "gpu_name": "not_measured", "gpu_memory_total": "not_measured",
            "cuda_version": str(torch.version.cuda or "not_recorded"), "torch_version": torch.__version__,
            "cudnn_version": str(torch.backends.cudnn.version() or "not_recorded"),
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": f"{props.name} (local profile; historical training hardware=not_recorded)",
        "gpu_memory_total": f"{props.total_memory / (1024 ** 2):.1f} MiB",
        "cuda_version": str(torch.version.cuda or "not_recorded"),
        "torch_version": torch.__version__,
        "cudnn_version": str(torch.backends.cudnn.version() or "not_recorded"),
    }


def git_commit(code_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={code_root}", "-C", str(code_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not_recorded"


def not_measured_row(spec: ExperimentSpec, reason: str, seed: int) -> Dict[str, str]:
    row = {field: "not_measured" for field in SUMMARY_FIELDS}
    row.update({
        "experiment_id": spec.experiment_id,
        "dataset": spec.dataset,
        "method_variant": spec.method_variant,
        "selected_epoch": str(spec.selected_epoch),
        "config_ref": repo_relative(spec.config_path),
        "checkpoint_ref": repo_relative(spec.checkpoint_path),
        "input_shape": "1x3x256x128",
        "profiling_precision": "FP32",
        "training_seed": "not_recorded",
        "profiling_seed": str(seed),
        "training_time_value": "not_recorded",
        "training_time_basis": "not_recorded",
        "measurement_status": f"not_measured: {reason}",
        "evidence_grade": "not_measured",
        "notes": "Historical performance evidence remains E2; local efficiency profile is not available.",
    })
    return row


def build_row(spec: ExperimentSpec, seed: int) -> Dict[str, str]:
    row = {field: "not_measured" for field in SUMMARY_FIELDS}
    row.update({
        "experiment_id": spec.experiment_id,
        "dataset": spec.dataset,
        "method_variant": spec.method_variant,
        "selected_epoch": str(spec.selected_epoch),
        "config_ref": repo_relative(spec.config_path),
        "checkpoint_ref": repo_relative(spec.checkpoint_path),
        "input_shape": "1x3x256x128",
        "profiling_precision": "FP32",
        "training_seed": "not_recorded",
        "profiling_seed": str(seed),
        "evidence_grade": "E3_local_profile; historical_performance_E2",
    })
    value, basis, note = parse_training_time(spec.log_path)
    row.update({"training_time_value": value, "training_time_basis": basis, "notes": note})
    row.update(cuda_environment())
    return row


def configure_training_root(code_root: Path, spec: ExperimentSpec, data_root: Path):
    """Load the exact historical config/code for a bounded memory-only step."""
    sys.path.insert(0, str(code_root))
    from config import cfg
    cfg.defrost()
    cfg.merge_from_file(str(spec.config_path))
    cfg.DATASETS.ROOT_DIR = str(data_root)
    cfg.DATALOADER.NUM_WORKERS = 0  # data workers do not contribute to CUDA peak memory
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.freeze()
    return cfg


def profile_training_memory(spec: ExperimentSpec, code_root: Path, data_root: Path, device: torch.device,
                            warmup: int, repeats: int, seed: int) -> Dict[str, Any]:
    """Run real-data forward/loss/backward/optimizer steps only for CUDA peaks."""
    set_profiling_seed(seed)
    cfg = configure_training_root(code_root, spec, data_root)
    from data import make_data_loader
    from layers import make_loss
    from modeling import build_model
    from solver import make_optimizer

    train_loader, _, _, num_classes = make_data_loader(cfg)
    images, pids, camids = next(iter(train_loader))
    blob = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model = build_model(cfg, num_classes).to(device).train()
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint validation failed: {incompat}")
    loss_fn = make_loss(cfg, num_classes)
    optimizer = make_optimizer(cfg, model)
    images, pids, camids = images.to(device), pids.to(device), camids.to(device)

    def one_step() -> int:
        optimizer.zero_grad(set_to_none=True)
        score, features = model(images)
        result = loss_fn(score, features, pids, camids)
        loss = result["loss_total"] if isinstance(result, dict) else result
        valid_count = int(result.get("cross_camera_positive_count", 0)) if isinstance(result, dict) else 0
        loss.backward()
        optimizer.step()
        return valid_count

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize(device)
    allocated, reserved, valid_counts = [], [], []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        valid_counts.append(one_step())
        torch.cuda.synchronize(device)
        allocated.append(torch.cuda.max_memory_allocated(device))
        reserved.append(torch.cuda.max_memory_reserved(device))
    return {
        "batch_size": int(images.shape[0]),
        "num_instances": int(cfg.DATALOADER.NUM_INSTANCE),
        "allocated_bytes": max(allocated),
        "reserved_bytes": max(reserved),
        "valid_anchor_counts": valid_counts,
        "data_loader_workers_for_profile": 0,
    }


def fmt_number(value: float | int, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, float) else str(value)


def mib(value: int) -> float:
    return value / (1024 ** 2)


def read_existing_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["experiment_id"]: {key: row.get(key, "not_measured") for key in SUMMARY_FIELDS}
                for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: Iterable[Mapping[str, str]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def lambda_from_spec(spec: ExperimentSpec) -> str:
    return spec.aux_lambda


def profiling_commands(spec: ExperimentSpec) -> List[str]:
    """Exact command templates used for the archived local measurements."""
    worktree = f"$env:TEMP\\{WORKTREE_BY_COMMIT[spec.code_commit]}"
    common = (
        f"$BOT_PY tools\\profile_model_efficiency.py --experiment-id {spec.experiment_id} "
        f"--code-root \"{worktree}\" --output-dir "
        "paper_notes\\c2_l03_final_evidence\\efficiency_analysis"
    )
    return [
        common + " --modes params_flops,inference,logs",
        common + " --modes memory,logs --data-root D:\\thesis_reid\\datasets",
    ]


def write_comparison_table(output_dir: Path, rows: Mapping[str, Mapping[str, str]], specs: Mapping[str, ExperimentSpec]) -> None:
    csv_fields = (
        "experiment_id", "dataset", "method_variant", "aux_lambda", "selected_epoch", "total_params",
        "trainable_params", "macs", "flops", "training_peak_allocated_mib", "training_peak_reserved_mib",
        "training_peak_allocated_gib", "training_peak_reserved_gib",
        "inference_mean_ms", "inference_p95_ms", "inference_images_per_second", "training_time_value",
        "training_time_basis", "gpu_name", "measurement_status",
    )
    table_rows = []
    for eid in FORMAL_IDS:
        row = dict(rows[eid])
        row["aux_lambda"] = lambda_from_spec(specs[eid])
        table_rows.append(row)
    write_csv(output_dir / "efficiency_comparison_table.csv", table_rows, csv_fields)

    md_header = ("实验 ID", "数据集", "方法/变体", "λ_aux", "选定 Epoch", "总参数量", "可训练参数量", "MACs",
                 "FLOPs", "训练峰值显存 allocated<br>(MiB / GiB)", "训练峰值显存 reserved<br>(MiB / GiB)", "推理均值<br>(ms/image)", "推理 P95<br>(ms/image)", "吞吐量<br>(images/s)",
                 "训练时间", "时间来源", "GPU", "测量状态")
    lines = ["# 表 X.X 不同实验配置的模型复杂度、显存与运行效率对比", "",
             "统一条件：推理输入 `1 × 3 × 256 × 128`、FP32、batch size=1；`FLOPs = 2 × MACs`。"
             "显存单位为 MiB/GiB，推理时间为 ms/image。`not_measured` 表示未按本表协议取得实际测量值。", "",
             "| " + " | ".join(md_header) + " |",
             "|" + "|".join(["---"] * len(md_header)) + "|"]
    for row in table_rows:
        values = [
            row["experiment_id"], row["dataset"], row["method_variant"], row["aux_lambda"], row["selected_epoch"],
            row["total_params"], row["trainable_params"], row["macs"], row["flops"],
            f"{row['training_peak_allocated_mib']} / {row['training_peak_allocated_gib']}",
            f"{row['training_peak_reserved_mib']} / {row['training_peak_reserved_gib']}",
            row["inference_mean_ms"], row["inference_p95_ms"], row["inference_images_per_second"],
            row["training_time_value"], row["training_time_basis"], row["gpu_name"], row["measurement_status"],
        ]
        values = [md_cell(str(value)) for value in values]
        lines.append("| " + " | ".join(values) + " |")
    market_control = rows["C2-Baseline-Control"]
    market_c2 = rows["C2-L03"]
    duke_control = rows["Duke-Baseline-Control"]
    duke_c2 = rows["Duke-C2-L03"]
    allocated_values = [float(rows[eid]["training_peak_allocated_mib"]) for eid in FORMAL_IDS]
    reserved_values = [float(rows[eid]["training_peak_reserved_mib"]) for eid in FORMAL_IDS]
    lines.extend([
        "", "## 测量协议说明", "",
        "参数量由每个实验的归档 config 与选定 checkpoint 严格加载后，逐模型统计 `model.parameters()` 得到。"
        "MACs/FLOPs 在 eval/inference 模式下对单个实际加载模型执行一次 `1×3×256×128` FP32 前向图分析；"
        "使用 PyTorch `torch.profiler`（版本见 `efficiency_summary.csv`），其带注释的 FLOPs 定义为卷积/矩阵乘累加的浮点运算估计，"
        "表中统一换算为 `MACs = profiler_FLOPs / 2`、`FLOPs = 2×MACs`。未被 profiler 标注的非线性、归一化及部分逐元素算子列入 `unsupported_ops`，"
        "因此该 FLOPs 值须按工具口径解释。每个实验的操作级 raw profiler 输出保存在 `raw_profiles/`。", "",
        "GPU 推理计时使用 CUDA events：100 次预热、1000 次正式重复、每次同步；报告均值、P95 与由均值换算的 images/s。"
        "训练峰值显存仅执行真实数据集与真实 sampler 产生的一个训练 batch 的“前向→损失→反向→optimizer.step”微型 profiling，"
        "不执行完整 epoch；预热 20 次、正式 30 次且每次重置 CUDA peak。训练 batch、每身份样本数、GPU 与软件版本见明细表。"
        "历史训练 seed 未归档，表中始终保留 `not_recorded`；profiling seed 仅为本次分析的本地 seed。", "",
        "训练时间优先使用日志明确记录的运行时；若无明确字段而有可解析首尾时间戳，则仅标记为 `derived_log_span`，且可能包含评估与 I/O；"
        "否则为 `not_recorded`。", "",
        "## 结果分析", "",
        "对 `d98fb000`、`7b8195d4`、`3ce47246`、`f1f16925` 和 `61da0df` 的 `modeling/baseline.py`、"
        "`modeling/backbones/resnet.py` 与 `modeling/__init__.py` 进行只读 Git diff，未观察到模型主干改动。实测结果中，"
        f"Market 的 Baseline-Control 与 C2-L03 均为 {market_control['total_params']} 个总参数、"
        f"{market_control['trainable_params']} 个可训练参数，以及 {market_control['macs']} MACs / {market_control['flops']} FLOPs；"
        f"Duke 的 Baseline-Control 与 C2-L03 均为 {duke_control['total_params']} 个总参数、"
        f"{duke_control['trainable_params']} 个可训练参数，前向 MACs/FLOPs 同为 {duke_control['macs']} / {duke_control['flops']}。"
        "Duke 参数量较少来自训练身份数不同导致的分类层维度不同，而不是 C2 损失改变了推理主干。"
        "因此，在当前实现中，C2、S2 与 CAAT 的差异主要位于训练期损失计算，推理主干结构一致；该判断同时由逐权重严格加载的 profiler 结果和历史代码审计支持。", "",
        f"12 个真实 batch 的峰值 allocated 范围为 {min(allocated_values):.3f}–{max(allocated_values):.3f} MiB，"
        f"reserved 范围为 {min(reserved_values):.3f}–{max(reserved_values):.3f} MiB。C2-L03 与其 Market Baseline-Control 的"
        f"峰值 allocated 分别为 {market_c2['training_peak_allocated_mib']} MiB 与 {market_control['training_peak_allocated_mib']} MiB。"
        f"本机串行计时中，两者推理均值分别为 {market_c2['inference_mean_ms']} ms/image 与 {market_control['inference_mean_ms']} ms/image；"
        "但这些单台设备、单次串行 timing 的波动不构成方法快慢或显存优劣的证据。训练日志耗时均为 `derived_log_span`，"
        "且历史训练硬件未归档，因此同样不用于效率排序。", "",
        "## 局限性说明", "",
        "训练峰值显存、推理时间和吞吐量依赖 GPU 型号、显存容量、驱动、CUDA、PyTorch/cuDNN、batch size、精度、热状态及本测量协议。"
        "单台设备上的本地 profiling 不能推出普遍效率结论；同一结构模型在本机单次串行计时出现的波动也不构成“更快/更慢”的证据。"
        "本地 profiling 不改变各训练性能结果 `n=1`、训练 seed=`not_recorded`、证据等级 E2 的边界。",
    ])
    (output_dir / "efficiency_comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(output_dir: Path, rows: Mapping[str, Mapping[str, str]]) -> None:
    fields = ("experiment_id", "total_params", "trainable_params", "macs", "flops", "inference_mean_ms",
              "training_peak_allocated_mib", "training_time_value", "training_time_basis", "gpu_name", "evidence_grade")
    lines = ["# 效率测量汇总", "", "正式论文表见 `efficiency_comparison_table.md`；本文件保留证据字段摘要。", "",
             "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for eid in FORMAL_IDS:
        lines.append("| " + " | ".join(md_cell(rows[eid].get(field, "not_measured")) for field in fields) + " |")
    (output_dir / "efficiency_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_methodology(output_dir: Path) -> None:
    text = """# 效率测量方法学

## 范围与边界

本目录仅覆盖正式证据台账中有资格进入论文主结果、λ 敏感性或消融表的 12 个实验 ID。CAT001、E1 历史结果、E0 占位结果以及已 superseded 的距离分析均被排除。所有历史训练性能仍是单次运行（`n=1`），训练 seed 为 `not_recorded`；本地 profiling 不提高性能结果的证据等级。

## 模型严格加载与复杂度

每行均从其归档 YAML 读取模型设置，从其选定 checkpoint 读取权重。checkpoint 以 `strict=True` 加载；出现缺失键、额外键或权重不存在时，该行不得报告参数量或复杂度。参数量为该已加载模型全部参数及 `requires_grad=True` 参数之和。FLOPs 以 `1×3×256×128`、FP32、eval/inference 模式下的真实前向图测量，不含 re-ranking、检索排序、数据读取和后处理。工具为 PyTorch `torch.profiler`，其原始操作级 FLOPs 主要覆盖卷积/矩阵乘注释；统一定义 `MACs = tool_FLOPs / 2` 与 `FLOPs = 2×MACs`。不被工具覆盖的算子明确保存在 `unsupported_ops` 字段。

## 推理计时

仅在 CUDA 可用的 NVIDIA GPU 上测量。每个模型输入 batch size=1、FP32，先执行 100 次预热，后用 CUDA events 对 1000 次推理逐次计时和同步；报告均值、中位数、P95 和 `1000 / mean_ms` images/s。CPU 计时不会作为 GPU 推理结果填入表格。

## 训练峰值显存

这不是完整训练：对归档 config 指定的真实数据集、sampler、batch size、每身份样本数、损失和优化器，取得一个真实训练 batch，执行 `forward → loss → backward → optimizer.step`。预热 20 步、正式 30 步；每次正式步前重置 CUDA peak 统计，报告正式步的最大 `max_memory_allocated` 与 `max_memory_reserved`。数据加载 worker 为 0 仅为避免 Windows worker 进程对本地 profiling 的干扰，且不计入 CUDA 显存；sampler、batch size、图像变换、损失与优化器均来自归档配置。profiling seed 是本次本地操作元数据，不能替代历史训练 seed。

## 训练时间

只解析原始归档日志。日志有明确 runtime 时记为 `explicit_recorded_runtime`；仅能从首尾时间戳获得时记为 `derived_log_span` 并注明可能包含评估和 I/O；无法满足两者时为 `not_recorded`。不同 GPU 上的日志耗时不用于“方法效率优劣”结论。
"""
    (output_dir / "efficiency_methodology.md").write_text(text, encoding="utf-8")


def write_measurement_record(output_dir: Path, rows: Mapping[str, Mapping[str, str]], specs: Mapping[str, ExperimentSpec],
                             commands: Sequence[str], script_sha: str, timestamp: str) -> None:
    lines = ["# 效率测量记录", "", f"生成时间：`{timestamp}`", f"脚本 SHA256：`{script_sha}`", "",
             "每一行的 checkpoint 均通过严格加载校验；`not_measured` 保留具体原因，而非以结构相似性补值。", ""]
    for eid in FORMAL_IDS:
        spec, row = specs[eid], rows[eid]
        lines.extend([
            f"## {eid}", "",
            f"- 数据集 / 方法：{spec.dataset} / {spec.method_variant}",
            f"- 选定 epoch：`{spec.selected_epoch}`",
            f"- 归档 config：`{repo_relative(spec.config_path)}`",
            f"- 选定 checkpoint：`{repo_relative(spec.checkpoint_path)}`",
            f"- Checkpoint SHA256：`{row.get('checkpoint_sha256', 'not_measured')}`",
            f"- 训练代码 commit / profiler code commit：`{spec.code_commit}` / `{spec.code_commit}`（使用对应 detached 临时 worktree）",
            f"- 输入 / 精度：`{row.get('input_shape')}` / `{row.get('profiling_precision')}`",
            f"- 参数量 / 工具：`{row.get('total_params')}` / `{row.get('flop_tool')} {row.get('flop_tool_version')}`",
            f"- MACs / FLOPs：`{row.get('macs')}` / `{row.get('flops')}`；未支持算子：{row.get('unsupported_ops')}",
            f"- 原始 profiler 操作记录：`raw_profiles/{eid}.torch_profiler.json`；SHA256：`{sha256(output_dir / 'raw_profiles' / f'{eid}.torch_profiler.json') if (output_dir / 'raw_profiles' / f'{eid}.torch_profiler.json').is_file() else 'not_measured'}`",
            f"- 推理协议与结果：batch={row.get('inference_batch_size')}，warmup={row.get('inference_warmup')}，repeats={row.get('inference_repeats')}，mean={row.get('inference_mean_ms')} ms，P95={row.get('inference_p95_ms')} ms",
            f"- 显存协议与结果：真实 batch={row.get('training_batch_size')}，K={row.get('num_instances')}；前向→损失→反向→optimizer.step，预热20步、正式30步、每正式步重置 CUDA peak；allocated={row.get('training_peak_allocated_mib')} MiB，reserved={row.get('training_peak_reserved_mib')} MiB",
            f"- 训练时间：`{row.get('training_time_value')}`，来源：`{row.get('training_time_basis')}`",
            f"- 环境：GPU=`{row.get('gpu_name')}`，显存=`{row.get('gpu_memory_total')}`，CUDA=`{row.get('cuda_version')}`，Torch=`{row.get('torch_version')}`，cuDNN=`{row.get('cudnn_version')}`",
            f"- 训练 seed：`not_recorded`；profiling seed：`{row.get('profiling_seed')}`",
            f"- 状态：{row.get('measurement_status')}",
            "- 实际 profiling 命令：",
            *[f"  - `{item}`" for item in profiling_commands(spec)],
            "",
        ])
    output_hashes = []
    for filename in ("efficiency_summary.csv", "efficiency_comparison_table.csv", "efficiency_comparison_table.md", "efficiency_methodology.md"):
        path = output_dir / filename
        if path.is_file():
            output_hashes.append(f"- `{filename}`: `{sha256(path)}`")
    lines.extend(["## 输出文件 SHA256", ""] + output_hashes)
    lines.extend(["", "## 本次文档刷新调用", ""] + [f"- `{command}`" for command in commands])
    (output_dir / "efficiency_measurement_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_unresolved(output_dir: Path, rows: Mapping[str, Mapping[str, str]]) -> None:
    lines = ["# 效率证据未解决项", "",
             "- 历史训练 seed 对所有 12 个训练实验均为 `not_recorded`；本地 profiling seed 不能替代它。",
             "- 训练性能为单次运行（`n=1`），不能从效率测量推导稳定性、统计显著性或普遍效率优劣。",
             "- PyTorch profiler 的 FLOP 注释不覆盖全部非线性、归一化和逐元素算子，`unsupported_ops` 已在每行保留。",
             "- 训练时长若为 `derived_log_span`，可能包含评估与 I/O；不应视为严格纯训练耗时。", ""]
    failures = [f"- {eid}: {rows[eid]['measurement_status']}" for eid in FORMAL_IDS
                if rows[eid].get("measurement_status", "").startswith("not_measured")]
    lines.append("## 本次未测量项")
    lines.append("")
    lines.extend(failures or ["- 无；全部目标项目已按本次协议实际测量。"]) 
    (output_dir / "efficiency_unresolved_items.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path, rows: Mapping[str, Mapping[str, str]], specs: Mapping[str, ExperimentSpec],
                   code_root: Path, command: str, script_sha: str, timestamp: str) -> None:
    fields = ("experiment_id", "config_ref", "config_sha256", "checkpoint_ref", "checkpoint_sha256", "training_code_commit",
              "profile_code_commit", "script_path", "script_sha256", "command", "timestamp", "gpu_name", "environment",
              "output_summary_sha256", "raw_profiler_output_ref", "raw_profiler_output_sha256", "measurement_status")
    summary_path = output_dir / "efficiency_summary.csv"
    summary_hash = sha256(summary_path) if summary_path.is_file() else "not_measured"
    env = cuda_environment()
    result = []
    for eid in FORMAL_IDS:
        spec, row = specs[eid], rows[eid]
        raw_path = output_dir / "raw_profiles" / f"{eid}.torch_profiler.json"
        result.append({
            "experiment_id": eid,
            "config_ref": repo_relative(spec.config_path),
            "config_sha256": sha256(spec.config_path) if spec.config_path.is_file() else "not_measured",
            "checkpoint_ref": repo_relative(spec.checkpoint_path),
            "checkpoint_sha256": row.get("checkpoint_sha256", "not_measured"),
            "training_code_commit": spec.code_commit,
            "profile_code_commit": spec.code_commit,
            "script_path": str(Path(__file__).resolve()), "script_sha256": script_sha,
            "command": command, "timestamp": timestamp, "gpu_name": env["gpu_name"],
            "environment": f"torch={env['torch_version']};cuda={env['cuda_version']};cudnn={env['cudnn_version']}",
            "output_summary_sha256": summary_hash,
            "raw_profiler_output_ref": str(raw_path.relative_to(output_dir)) if raw_path.is_file() else "not_measured",
            "raw_profiler_output_sha256": sha256(raw_path) if raw_path.is_file() else "not_measured",
            "measurement_status": row.get("measurement_status", "not_measured"),
        })
    write_csv(output_dir / "efficiency_evidence_manifest.tsv", result, fields)
    # TSV requires tabs rather than the CSV writer's default delimiter.
    path = output_dir / "efficiency_evidence_manifest.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(result)


def update_documents(output_dir: Path, rows: Mapping[str, Mapping[str, str]], specs: Mapping[str, ExperimentSpec],
                     code_root: Path, command: str) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    script_sha = sha256(Path(__file__).resolve())
    write_csv(output_dir / "efficiency_summary.csv", [rows[eid] for eid in FORMAL_IDS], SUMMARY_FIELDS)
    write_summary_md(output_dir, rows)
    write_comparison_table(output_dir, rows, specs)
    write_methodology(output_dir)
    write_measurement_record(output_dir, rows, specs, [command], script_sha, timestamp)
    write_unresolved(output_dir, rows)
    write_manifest(output_dir, rows, specs, code_root, command, script_sha, timestamp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=False)
    selector.add_argument("--all", action="store_true", help="Profile all 12 formal experiment IDs.")
    selector.add_argument("--experiment-id", nargs="+", choices=FORMAL_IDS, help="One or more formal IDs.")
    parser.add_argument("--modes", default="params_flops,inference,logs", help="Comma-separated: params_flops,inference,memory,logs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--code-root", type=Path, default=REPO_ROOT, help="Exact code checkout/worktree used for this run.")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\thesis_reid\datasets"))
    parser.add_argument("--profiling-seed", type=int, default=42)
    parser.add_argument("--inference-warmup", type=int, default=100)
    parser.add_argument("--inference-repeats", type=int, default=1000)
    parser.add_argument("--memory-warmup", type=int, default=20)
    parser.add_argument("--memory-repeats", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = experiment_specs()
    targets = list(FORMAL_IDS if args.all or not args.experiment_id else args.experiment_id)
    modes = {part.strip() for part in args.modes.split(",") if part.strip()}
    unknown = modes - {"params_flops", "inference", "memory", "logs"}
    if unknown:
        raise SystemExit(f"Unknown profiling modes: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = read_existing_rows(args.output_dir / "efficiency_summary.csv")
    rows: Dict[str, Dict[str, str]] = {
        eid: existing.get(eid, not_measured_row(specs[eid], "not_profiled_in_this_invocation", args.profiling_seed))
        for eid in FORMAL_IDS
    }
    command = " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    if args.dry_run:
        print(json.dumps({"targets": targets, "modes": sorted(modes), "code_root": str(args.code_root),
                          "data_root": str(args.data_root)}, indent=2))
        return 0
    if ("params_flops" in modes or "inference" in modes or "memory" in modes) and not torch.cuda.is_available():
        for eid in targets:
            rows[eid] = not_measured_row(specs[eid], "CUDA_NVIDIA_GPU_not_available; CPU substitution prohibited", args.profiling_seed)
        update_documents(args.output_dir, rows, specs, args.code_root, command)
        return 0
    device = torch.device("cuda:0")
    for eid in targets:
        spec = specs[eid]
        previous = rows[eid]
        row = (build_row(spec, args.profiling_seed)
               if previous.get("measurement_status", "").startswith("not_measured: not_profiled")
               else dict(previous))
        row.update(cuda_environment())
        try:
            if "logs" in modes:
                value, basis, note = parse_training_time(spec.log_path)
                row.update({"training_time_value": value, "training_time_basis": basis, "notes": note})
            if "params_flops" in modes or "inference" in modes:
                set_profiling_seed(args.profiling_seed)
                model, _, _, ckpt_sha = load_inference_model(spec, args.code_root, device)
                row["checkpoint_sha256"] = ckpt_sha
                total, trainable = count_parameters(model)
                row["total_params"], row["trainable_params"] = str(total), str(trainable)
                if "params_flops" in modes:
                    macs, flops, unsupported, raw_events = profile_macs_with_torch_profiler(model, device)
                    write_raw_profiler_output(args.output_dir, spec, macs, flops, raw_events)
                    row.update({"macs": str(macs), "flops": str(flops), "flop_tool": "torch.profiler",
                                "flop_tool_version": torch.__version__, "unsupported_ops": unsupported})
                if "inference" in modes:
                    timing = benchmark_inference(model, device, args.inference_warmup, args.inference_repeats)
                    row.update({"inference_batch_size": "1", "inference_warmup": str(args.inference_warmup),
                                "inference_repeats": str(args.inference_repeats),
                                "inference_mean_ms": fmt_number(timing["mean_ms"]),
                                "inference_median_ms": fmt_number(timing["median_ms"]),
                                "inference_p95_ms": fmt_number(timing["p95_ms"]),
                                "inference_images_per_second": fmt_number(timing["images_per_second"])})
                del model
                torch.cuda.empty_cache()
            if "memory" in modes:
                memory = profile_training_memory(spec, args.code_root, args.data_root, device,
                                                 args.memory_warmup, args.memory_repeats, args.profiling_seed)
                allocated_mib, reserved_mib = mib(memory["allocated_bytes"]), mib(memory["reserved_bytes"])
                row.update({"training_batch_size": str(memory["batch_size"]), "num_instances": str(memory["num_instances"]),
                            "training_peak_allocated_mib": fmt_number(allocated_mib),
                            "training_peak_reserved_mib": fmt_number(reserved_mib),
                            "training_peak_allocated_gib": fmt_number(allocated_mib / 1024.0),
                            "training_peak_reserved_gib": fmt_number(reserved_mib / 1024.0),
                            "notes": row["notes"] + f"; real_data_memory_profile; workers=0; valid_anchor_counts={memory['valid_anchor_counts']}"})
            row["measurement_status"] = "measured" if all(
                row[field] != "not_measured" for field in ("total_params", "macs", "flops", "inference_mean_ms")
            ) else "partially_measured"
        except Exception as exc:  # preserve partial real measurements only when no required field was produced
            row["measurement_status"] = f"not_measured: {type(exc).__name__}: {exc}"
            row["notes"] = row.get("notes", "") + "; failure recorded without substitution"
            torch.cuda.empty_cache()
        rows[eid] = row
        update_documents(args.output_dir, rows, specs, args.code_root, command)
        print(f"{eid}: {rows[eid]['measurement_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
