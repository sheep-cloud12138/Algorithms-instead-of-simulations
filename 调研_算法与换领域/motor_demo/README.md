# motor_demo

2D 非线性静磁 + 绕组接法扫描，用来验证"电机绕组切换能不能纯仿真、值不值得做"。

## 结论

- 能仿真：44,800 单元细网格 118 s/工况，气隙 0.594 T。
- 包络不重合：串联低速占优，并联/三角形在 **1.9~2.2 倍基速**后反超。
- 收益取决于 `I_ch = λ/L`：高电感机器高速铜损低 2~3 倍；低电感机器只是把硬转速墙往后推、转矩减半。

详细数据见 `转矩包络验证.md`。

## 文件

| 文件 | 作用 |
|---|---|
| `demo_magnetostatic.py` | 单工况静磁，出磁场图 |
| `motor_sweep.py` | 参数化求解器（转子角、接法、电流） |
| `run_sweep.py` | 3 接法 × 12 转子位置扫描，出 `sweep_3x12.csv` |
| `envelope_two_cases.py` | 电压/电流受限包络 + 弱磁，两种电感对比 |
| `envelope.py` | 单电感版本的包络 |
| `转矩包络验证.md` | 包络结果、判断、局限 |
| `参数判定_demo电机.md` | λ/L/R、I_ch/I_max、立项判据 |
| `params_demo.py` | 参数提取脚本 |
| `仿真可行性实测.md` | 单工况可行性与算力账 |

## 跑

```powershell
cd D:\Desktop\算法代替仿真\调研_算法与换领域\motor_demo
python demo_magnetostatic.py          # 单工况，约 2 min（细网格）
python run_sweep.py                   # 36 工况，约 20 s（6 进程）
python envelope_two_cases.py          # 包络图
```

## 注意

这是可行性 demo，不是验证过的设计工具：转矩未网格收敛、槽口有尖角奇异、70% 脉动未优化、非线性迭代停在 rel≈1e-4。
