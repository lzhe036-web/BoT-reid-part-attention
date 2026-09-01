# 固定样本图像面板

本目录的 `clear`、`occluded`、`misaligned`、`side_view`、`back_view`、`blurred` 是为盲标注后的展示面板保留的目录。Market1501 不提供这些官方类别；标注必须先于读取门控权重，且写入 `manifests/image_type_annotations.tsv`。当前工具不会把结构删除的 scale 写成 0；它必须显示为 `excluded`。
