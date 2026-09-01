# 固定样本盲标注说明

只能在未展示任何模型权重、检索指标或预测结果的情况下查看图片后填写
`image_type_annotations.tsv`。允许同一 `stable_sample_key` 多行，从而表达多标签。
类别限定为：`clear`、`occluded`、`misaligned`、`side_view`、`back_view`、`blurred`。
每个类别按固定 manifest 的 `selection_hash` 升序选取，不能依据任何门控统计补选图片。
