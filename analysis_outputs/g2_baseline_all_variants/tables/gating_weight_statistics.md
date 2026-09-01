# 门控权重统计（统一 G2 baseline）

native_applied_weight 为模型实际使用的 w；normalized_probability 为和为 1 的 p。excluded 表示结构未提供该分支，绝不等同于零。

| baseline | variant | gate_input | active_scales | weight | statistic_type | status | native_weight_sum | count | mean | std | min | max | median | q25 | q75 | dominant_ratio | ci95_low | ci95_high | baseline_mean | delta_vs_g2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G2-global-local | G2-global-local | concat_global_local | 2,4,6 | w2 | native_applied_weight | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-global-local | concat_global_local | 2,4,6 | w2 | normalized_probability | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-global-local | concat_global_local | 2,4,6 | w4 | native_applied_weight | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-global-local | concat_global_local | 2,4,6 | w4 | normalized_probability | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-global-local | concat_global_local | 2,4,6 | w6 | native_applied_weight | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-global-local | concat_global_local | 2,4,6 | w6 | normalized_probability | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G1 | global | 2,4,6 | w2 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G1 | global | 2,4,6 | w2 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G1 | global | 2,4,6 | w4 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G1 | global | 2,4,6 | w4 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G1 | global | 2,4,6 | w6 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G1 | global | 2,4,6 | w6 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-local-only | concat_local | 2,4,6 | w2 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-local-only | concat_local | 2,4,6 | w2 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-local-only | concat_local | 2,4,6 | w4 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-local-only | concat_local | 2,4,6 | w4 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-local-only | concat_local | 2,4,6 | w6 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-local-only | concat_local | 2,4,6 | w6 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z6 | concat_z2_z4 | 2,4 | w2 | native_applied_weight | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z6 | concat_z2_z4 | 2,4 | w2 | normalized_probability | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z6 | concat_z2_z4 | 2,4 | w4 | native_applied_weight | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z6 | concat_z2_z4 | 2,4 | w4 | normalized_probability | active_fixed_candidate_manifest_required_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z6 | concat_z2_z4 | 2,4 | w6 | native_applied_weight | excluded | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z6 | concat_z2_z4 | 2,4 | w6 | normalized_probability | excluded | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z4 | concat_z2_z6 | 2,6 | w2 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z4 | concat_z2_z6 | 2,6 | w2 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z4 | concat_z2_z6 | 2,6 | w4 | native_applied_weight | excluded | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z4 | concat_z2_z6 | 2,6 | w4 | normalized_probability | excluded | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z4 | concat_z2_z6 | 2,6 | w6 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z4 | concat_z2_z6 | 2,6 | w6 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z2 | concat_z4_z6 | 4,6 | w2 | native_applied_weight | excluded | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z2 | concat_z4_z6 | 4,6 | w2 | normalized_probability | excluded | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z2 | concat_z4_z6 | 4,6 | w4 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z2 | concat_z4_z6 | 4,6 | w4 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z2 | concat_z4_z6 | 4,6 | w6 | native_applied_weight | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| G2-global-local | G2-without-z2 | concat_z4_z6 | 4,6 | w6 | normalized_probability | active_missing_formal_evidence_unavailable | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
