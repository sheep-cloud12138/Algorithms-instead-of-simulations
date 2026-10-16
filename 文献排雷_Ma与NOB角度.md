# 文献排雷：把"相似律破缺"当学习对象（Ma / NOB 角度）

> 检索日期 2026-10-16。渠道：OpenAlex 标题+摘要精确检索（主力）、Semantic Scholar bulk、arXiv API。
> S2 单篇/批量接口本轮持续 429 或返回 0，只作旁证。

## 结论

**"相似律破缺量本身"作为机器学习/代理模型的学习对象，没有先例。这条路直接做。**

| 检索式（OpenAlex, title+abstract） | 命中 | 判断 |
|---|---|---|
| `non-Oberbeck-Boussinesq` | **135** | 物理侧很成熟 —— 这是"破缺"在传热界的正名 |
| `non-Oberbeck-Boussinesq` + `machine learning \| neural \| surrogate` | 3 | 逐条看过，全是关键词溢出，实际 ≈ **0** |
| `non-Boussinesq` + `neural` | **0** | 空白 |
| `Boussinesq` + `neural operator` | 3 | 唯一相关的就是 [2606.25259](https://arxiv.org/abs/2606.25259)；另两条是 Boussinesq **波**（海啸），不是对流 |
| `Boussinesq` + `surrogate model` | 21 | 绝大多数是波浪/地幔对流/润滑；对流相关的只有 POD+RNN 代理（IJCFD 2022），没有破缺角度 |
| `compressible` + `Rayleigh-Benard` + `machine learning` | **0** | 空白 |
| `low Mach number` + `natural convection` | 1 | 2026 EAAI 变压器数字孪生，顺带提了 low Mach，不是破缺研究 |
| `similarity breaking` + `convection` | **0** | 空白 |
| `Rayleigh-Benard` + `neural network` | 10 | 全是 PINN 重构/降尺度/纳米流体 ANN 拟合，无破缺角度 |

## 物理侧：破缺被研究得很透（这是我们的支点，不是撞车）

| 文献 | 为什么重要 |
|---|---|
| Ahlers et al. 2006, JFM, `10.1017/s0022112006002916` | NOB 效应对 Nu/Re 的经典理论与实验 |
| 2007 PRL `10.1103/physrevlett.98.054501` | **气体**的 NOB 效应 |
| 2009 JFM `10.1017/s0022112009008027` | 2D NOB Rayleigh–Bénard 的流场组织与温度剖面 |
| **2016 JFM `10.1017/jfm.2016.338`** | **理想气体、Pr=0.71、大温差**下的 NOB 流动反转 —— 与我们的可压缩算例物理设定几乎一致 |
| **2018 IJHMT `10.1016/j.ijheatmasstransfer.2018.04.135`** | "高效模拟 NOB 流动的数值方法" —— **IJHMT 收这个主题** |

合起来就是我们能站的位置：物理侧有公认的破缺现象、有明确的工程后果（Nu 偏差、流动反转），**但没人给它建过数据驱动代理**。

## 与 2606.25259 的区分

| 维度 | 2606.25259 | 本方案 |
|---|---|---|
| 框架 | 低保真 + NN 修正 | 不比框架（已撞车，论文里一句话带过） |
| 变化轴 | Ra（+ 边界复杂度） | **Ma 作为独立破缺群**（他们代码里 Ma 是 Ra 的幂律副产品，从未独立扫描） |
| 学习目标 | 提高场精度（MSE/SSIM） | **破缺量本身的结构与标度律** |
| 物理结论 | 无（纯方法学） | 有：修正项幅度按 Ma 的什么幂次长？弱破缺训练的修正能不能迁到强破缺？破缺量能不能加？ |

## 要拿下的两点

1. **贡献只写在破缺上。** 精度提升是副产品，不算贡献。论文的骨头是那三条标度/迁移/可加性判据；立住就是新东西。
2. **术语分清。** NOB 的经典定义是**变物性**（黏度/导热随温度变）；我们的 Ma 轴是**可压缩性**（压力功、黏性耗散、理想气体状态方程）。同属"偏离 Boussinesq"但机制不同，各自点名，不混用。

## 覆盖范围

检索的是 OpenAlex 标题+摘要字段。S2 限流、Google Scholar 不可达。这层覆盖能回答"有没有人做过"，后面盯 arXiv 新刊即可。