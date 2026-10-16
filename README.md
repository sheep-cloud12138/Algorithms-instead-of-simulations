# 代理模型代替仿真：Boussinesq → 可压缩自然对流的物理先验代理

用一个**可辨识的破缺参数** `ε = ΔT/T₀` 把「低保真（Boussinesq）解」和「高保真（可压缩）解」之间的差
写成闭式，并把它做成网络的输出结构 `D = ε·P(X)`，从而在少量数据下把小时级仿真压到毫秒级。

## 三句话结论

1. **律**：`R = ‖Y−X‖/‖Y‖ = A ε^p`，`p = 1.02 / 1.00 / 1.02 / 1.01`（u_x/u_y/θ/p，两个十年窗口），
   截距与 0 无法区分（u_x：+0.0003 ± 0.0047，0.06σ）。窗口收窄到 ε ≥ 0.8 时有效指数升到 1.23–1.32，
   正是理论里 `ε²` 项的指纹（`R = Aε(1+κε)`，κ = 0.42–0.83）。
2. **先验**：把 `ε` 从网络输入里拿掉、只让它出现在输出端（4 行代码），外推误差相对「朴素网络」下降约 1.6×，
   并且网络自己学出正的曲率 β = 0.20–0.29。它的价值不是内插更准，而是**单温差数据集可复用**。
3. **外部验证**：公开数据集（另一课题组、另一套生成器，ε ≡ 1.2）的 5 个划分、9600 个样本、38400 个
   relL2 值逐位复现（最大偏差 1.2e-7）；同一 BC 族内跨数据集的律可迁移，跨 BC 族的迁移失败——
   而失败的原因（A 是流动的泛函）本身是可以预测、可以验证的。

加速比（同一台机器实测）：求解 33.9 s/样本（中位）→ 推理 6.3 ms ⇒ **5.4×10³**（单样本）/ **6.9×10⁴**（批量）。

## 目录

| 路径 | 内容 |
|---|---|
| `论文/` | 英文论文正文、理论推导、图表 |
| `研究结果/` | 中文结果汇总、实验清单、所有图（PNG） |
| `code/` | 全部脚本：数据生成后处理、实验、画图、制表、审计 |
| `research/` | 文献调研、选题评审、公开数据集下载脚本与原始数据（大文件不入库） |

## 复现

环境：WSL2 Ubuntu-22.04。求解器用 dolfinx 0.7.3（`~/micromamba/envs/fx`），
网络用 PyTorch 2.10 + CUDA（`~/miniconda3/envs/agent`）。原始算例与中间结果放在 `~/nc_run/`：

```
~/nc_run/p2/pairs600_fixedp.npz     # 600 组主数据（4 通道，压力已按公开配方修正）
~/nc_run/paired/dataset/            # 20 配置 × 5 个 ε 的配对集（精确 ε-blind 对照）
~/nc_run/eps0/dataset/              # 27 组 ε→0 核验点
~/nc_run/water/dataset/             # Pr = 4.38 的 104 组（与空气逐位配对）
~/nc_run/cavity2/dataset/           # 2:1 几何 104 组
~/nc_run/pub/                       # 公开数据集 train / test_in_distribution / test_ood
~/nc_run/p7/*.json                  # 所有结果（论文里每个数字都能追到某个 JSON）
```

主要入口（都在 `code/`，用 `agent` 环境的 python 跑）：

```bash
python exp_eps_law.py        # 627 组上的律 + 二次项 + ε→0 截距
python exp_paired_final.py   # ε-blind 天花板 + ε 迁移矩阵 + 形状不变性
python exp_ml_final.py       # 统一消融 A/B/P1/P2/P3（3 种子）
python exp_fno_final.py      # FNO 基线（同协议）
python exp_lc_final.py       # 学习曲线
python exp_pod.py            # POD-ROM / 线性代理基线
python exp_shallow.py        # 岭回归 / 核岭回归基线
python exp_public_transfer.py, exp_public_legacy_transfer.py   # 跨数据集迁移
python exp_uhat_law.py       # θ 通道的 û 律（六个数据集）
python audit_v2.py           # 稳健性审计（指数窗口、κ、压力约定、û、σ 距离）
python make_tables.py        # 从 JSON 生成论文里所有表格
python make_figs.py; python make_figs2.py    # 所有图
```

## 分支

* `dev`：日常全部改动。
* `master`：只有当一项工作完成、且经过独立 sub-agent 复核通过后才推进。