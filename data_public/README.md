# data_public

本目录放论文可复现所需的数据。`.pt` 为 `torch.save` 张量，`.npz` 为 `numpy` 归档。

| 文件 | 大小 | 内容 |
|---|---|---|
| `pairs600_fixedp.npz` | 69 MB | 627 组配对解（Boussinesq `X` 与可压缩 `Y`），ε=0.02–1.9 |
| `test_in_distribution_X.pt` / `_Y.pt` | 各 13 MB | 分布内测试集 |
| `test_legacy_X.pt` / `_Y.pt` | 各 13 MB | 公开数据集（Menglik et al., ε=1.2）测试集副本 |
| `*_per_sample.csv` | 各 0.1 MB | 逐样本误差表 |
| `*_MANIFEST.json` | 各 0.3 MB | 样本清单与元数据 |

## pairs600_fixedp.npz

```python
import numpy as np, json
d = np.load("data_public/pairs600_fixedp.npz")
X    = d["X"]     # (600, 4, 64, 64) float32  Boussinesq 解
Y    = d["Y"]     # (600, 4, 64, 64) float32  可压缩解
meta = json.loads(str(d["meta"]))   # list[dict]，每项含 Ra, eps, dT, T0, nT, crel, tag
```

通道顺序与 `code/` 中的加载器一致。训练/测试划分见 `*_MANIFEST.json`。

原始逐算例网格文件与中间场（约 4 GB）不在仓库内。
