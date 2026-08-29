#!/usr/bin/env python
"""Immutable identities for the formally compared Dynamic Gating variants.

The profiles keep variant-specific names and validation contracts out of the
shared analyzer, finalizer, and recovery code.  They are deliberately data
only: changing a profile is a protocol change, not a way to reinterpret an
existing recorded run.
"""

from __future__ import absolute_import

from collections import namedtuple


G2DynamicGatingProfile = namedtuple("G2DynamicGatingProfile", (
    "key",
    "experiment_id",
    "expected_branch",
    "expected_parent_branch",
    "gating_input",
    "gating_input_semantics",
    "experiment_label",
    "method_variant",
    "method",
    "config_filename",
    "formal_result_filename",
    "formal_result_artifact_type",
    "analysis_directory_name",
    "analysis_manifest_filename",
    "artifact_prefix",
    "controller_blocks",
    "active_scales",
    "gate_weight_sum",
    "required_analysis_artifacts",
))


G2_GLOBAL_LOCAL_PROFILE = G2DynamicGatingProfile(
    key="g2_global_local",
    experiment_id="C2-L03-MGDG-G2-GL-T1-S42",
    expected_branch="codex/g2-global-local-gating",
    expected_parent_branch="exp/c2-l03-multi-granularity-dynamic-gating",
    gating_input="concat_global_local",
    gating_input_semantics="concat([g, z2, z4, z6])",
    experiment_label="G2 global-plus-local Dynamic Gating",
    method_variant="g2_global_local_per_sample_dynamic_gating",
    method="C2-L03 + G2 Dynamic Gating [g,z2,z4,z6] -> [w2,w4,w6]",
    config_filename=(
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
        "g2_global_local_autodl.yml"
    ),
    formal_result_filename="g2_formal_result.json",
    formal_result_artifact_type="g2_formal_result",
    analysis_directory_name="g2_gating_analysis",
    analysis_manifest_filename="g2_gating_analysis_manifest.json",
    artifact_prefix="g2",
    controller_blocks=("g", "z2", "z4", "z6"),
    active_scales=(2, 4, 6),
    gate_weight_sum=3.0,
    required_analysis_artifacts=(
        "controller_block_norms_csv",
        "controller_block_norms_png",
        "test_gate_samples_tsv",
        "test_weight_summary_csv",
        "test_weight_distribution_png",
        "dynamic_gating_summary_json",
    ),
)


G2_LOCAL_ONLY_PROFILE = G2DynamicGatingProfile(
    key="g2_local_only",
    experiment_id="C2-L03-MGDG-G2-LOCAL-T1-S42",
    expected_branch="codex/g2-local-only",
    expected_parent_branch="codex/g2-global-local-gating",
    gating_input="concat_local",
    gating_input_semantics="concat([z2,z4,z6])",
    experiment_label="G2-local-only Dynamic Gating",
    method_variant="g2_local_only_per_sample_dynamic_gating",
    method="G2-local-only Dynamic Gating [z2,z4,z6] -> [w2,w4,w6]",
    config_filename=(
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
        "g2_local_only_autodl.yml"
    ),
    formal_result_filename="g2_local_only_formal_result.json",
    formal_result_artifact_type="g2_local_only_formal_result",
    analysis_directory_name="g2_local_only_gating_analysis",
    analysis_manifest_filename="g2_local_only_gating_analysis_manifest.json",
    artifact_prefix="g2_local_only",
    controller_blocks=("z2", "z4", "z6"),
    active_scales=(2, 4, 6),
    gate_weight_sum=3.0,
    required_analysis_artifacts=(
        "controller_block_norms_csv",
        "controller_block_norms_png",
        "test_gate_samples_tsv",
        "test_weight_summary_csv",
        "test_weight_distribution_png",
        "test_weight_distribution_pdf",
        "dynamic_gating_summary_json",
    ),
)


G2_WITHOUT_Z6_PROFILE = G2DynamicGatingProfile(
    key="g2_without_z6",
    experiment_id="C2-L03-MGDG-G2-WITHOUT-Z6-T1-S42",
    expected_branch="codex/g2-without-z6",
    expected_parent_branch="codex/g2-local-only",
    gating_input="concat_z2_z4",
    gating_input_semantics="concat([z2,z4])",
    experiment_label="G2-without-z6 Dynamic Gating",
    method_variant="g2_without_z6_per_sample_dynamic_gating",
    method="G2-without-z6 Dynamic Gating [z2,z4] -> [w2,w4]",
    config_filename=(
        "softmax_triplet_c2_l03_multi_granularity_dynamic_gating_"
        "g2_without_z6_autodl.yml"
    ),
    formal_result_filename="g2_without_z6_formal_result.json",
    formal_result_artifact_type="g2_without_z6_formal_result",
    analysis_directory_name="g2_without_z6_gating_analysis",
    analysis_manifest_filename="g2_without_z6_gating_analysis_manifest.json",
    artifact_prefix="g2_without_z6",
    controller_blocks=("z2", "z4"),
    active_scales=(2, 4),
    gate_weight_sum=1.0,
    required_analysis_artifacts=(
        "controller_block_norms_csv",
        "controller_block_norms_png",
        "test_gate_samples_tsv",
        "test_weight_summary_csv",
        "test_weight_distribution_png",
        "test_weight_distribution_pdf",
        "dynamic_gating_summary_json",
    ),
)
