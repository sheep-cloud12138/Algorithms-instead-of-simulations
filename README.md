# 代理模型代替仿真：Boussinesq → 可压缩自然对流的物理先验代理

用一个**可辨识的破缺参数** `ε = ΔT/T₀` 把「低保真（Boussinesq）解」和「高保真（可压缩）解」之间的差
写成闭式，并把它做成网络的输出结构 `D = ε·P(X)`，从而在少量数据下把小时级仿真压到毫秒级。

## 三句话结论

1. **律**：`R = ‖Y−X‖/‖Y‖ ≈ A ε`，名义幂律指数接近 1；但 600 点核心只有 134 个壁面族，
   按族重采样后区间变宽，曲率幅度未分辨。窗口收窄到 ε ≥ 0.8 时有效指数升到 1.23–1.32，
   与 `ε²` 项一致（`R = Aε(1+κε)`）。
2. **先验**：把 `ε` 从网络输入里拿掉、只让它出现在输出端（4 行代码），外推误差相对「朴素网络」下降 1.61×，
   并且网络自己学出正的曲率 β = 0.22–0.42。样本少时它在分布内也赢（n ≤ 200，n=50 时好 1.4×，
   交叉点约 n ≈ 300），价值是**样本少时更省 + 单温差数据集可复用**。
3. **一致性检查**：公开数据集（另一课题组、另一套生成器，ε ≡ 1.2）的 5 个划分、9600 个样本、
   38400 个 relL2 值逐位复现（最大偏差 1.2e-7）；同一 BC 族内可复用，跨 BC 族的迁移失败。
   这会限定为已测族的一致性检查，不外推为通用跨族规律。

加速比（同一台机器实测）：单对 33.9 s → 11.7 s（**2.9×**，端到端）；20 流 × 5 个 ε 为 **13×**；
单流扫 100 个 ε 为 **2.3×10²**。单次推理 6.3 ms（GPU）/ 14.0 ms（CPU 单线程）。

## 目录

| 路径 | 内容 |
|---|---|
| `论文/` | 英文论文正文、理论推导、图表 |
| `研究结果/` | 中文结果汇总、实验清单、所有图（PNG） |
| `code/` | 全部脚本：数据生成后处理、实验、画图、制表、审计 |
| `research/` | 文献检索报告与原始索引、公开数据集下载脚本 `dl.py`、原始数据（大文件不入库） |
| `调研_算法与换领域/` | 早期「换方向」调研报告（未采用路线，留档） |

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
python audit_rigor_v2.py     # 目标网格不确定度、实际U-Net模板复评、种子CI和3D ROM审计
python make_tables.py        # 从 JSON 生成论文里所有表格
python make_figs.py; python make_figs2.py    # 所有图
```

## 分支

* `dev`：日常全部改动。
* `master`：只有当一项工作完成、且经过独立 sub-agent 复核通过后才推进。
