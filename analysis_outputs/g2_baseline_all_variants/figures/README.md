# 图表状态

只有同时满足以下条件时，才允许生成跨版本门控图：固定候选 manifest 已在读取任何权重前冻结；G2 和 comparator 的原始门控 TSV 均可校验；每一个固定样本均可跨版本配对。当前状态详见 `analysis_status.json`。工具不会用不同候选集、smoke 或其他分支的 TSV 生成替代图表。

成功配对时，每组 `g2_vs_*` 都会从对应的 `tables/paired_samples_g2_vs_*.csv` 生成 PNG 和 PDF（300 DPI）：归一化概率直方图、dominant K 比例、配对概率差和 dominant K 转移矩阵。`samples/` 仅用于盲标注后按固定 hash 顺序制作展示面板；未完成盲标注时不会输出挑选的图像面板。
