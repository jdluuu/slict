# A/B/C 本机实测记录

**历史实验说明：** 本文整帧回放采用 A=1、B/C=8 的内部迭代上限，属于不同计算预算，不能用作当前 A/B/C 统一单步关联调度的基线。当前代码固定三组每次一步，随后更新、去畸变和重新关联；使用方法见 [当前文档](solver_comparison.md)。以下测量值和原始命令保留用于追溯，不代表新的调度或 NTU VIRAL 测试结果。

2026-09-06，SLICT `noetic` (`0a79e975`)，AMD Ryzen 9 9950X（16 核 / 32 线程），Ubuntu 24.04.2，GCC 13.3，Eigen 3.4，Ceres 2.1 / SuiteSparse 7.6.1，SPLBATCH 0.1.1 (`e2892937`)。Release 构建全部通过，编译使用 `--parallel 16`。实现和复现入口见 [使用文档](solver_comparison.md)，可复查的汇总与环境清单保存在 [JSON 结果](solver_comparison_results.json)。

## 数值检查

数值测试和 97 个真实快照均通过。快照来自 R3LIVE `degenerate_seq_02.bag` 前 30 秒的 A 回放，跨度约 28.8 秒，每三个处理帧保存一次。该次 A 全程保有 LiDAR 约束。

| 检查 | 最大差异 |
| --- | ---: |
| A 与 B/C 的 H（相对） | 5.37e-16 |
| A 与 B/C 的梯度（相对） | 1.84e-12 |
| A 与 B/C 的初始代价（相对） | 2.35e-13 |
| B/C 求解状态，分别 1 和 8 次迭代 | 1.16e-10 |

状态差异取所有控制点平移、SO(3) log 旋转差、gyro/accel bias 的最大向量范数；这是同一优化问题的数值检查，未使用轨迹真值。独立单元测试还验证了右乘有限差分、先验导数、部分 Jacobian 请求，以及 Schur 先验与 QR 消元的二次代价变化。

## 同一冻结问题：一次内部迭代

97 个相同窗口，每组每个线程配置正式测量 5 轮，共 485 个样本。各后端独立进程、每轮全量预热一次、轮换执行次序。实际测量期间没有同时进行编译或 ROS 回放，CPU 亲和集合相同，BLAS 为单线程。每个窗口约 8,000 个 LiDAR 观测、约 73 个 IMU 观测、33–34 个控制点，先验通过同一快照提供。

表中单位为 ms，`均值 / P95`；时间包含重置、构造、求解和析构，不含公共最终代价检查、快照 I/O 和边缘化。

| 优化线程 | A 均值 / P95 | B 均值 / P95 | C 均值 / P95 | B/C 配对时间比中位数 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.35 / 17.76 | 35.14 / 36.17 | 9.73 / 9.94 | 3.61 |
| 4 | 7.85 / 8.44 | 31.08 / 32.28 | 5.94 / 6.60 | 5.28 |
| 8 | 6.58 / 7.73 | 29.75 / 30.90 | 5.07 / 5.55 | 5.93 |
| 16 | 6.36 / 7.09 | 28.83 / 30.25 | 4.92 / 5.36 | 5.86 |

A 的平均最终代价为 5.150323，B/C 为 5.167904；B/C 最大相对代价差 1.24e-13，全部解可用，无显著代价上升样本。A/C 配对时间比中位数分别为 1.79、1.33、1.28、1.30，但二者优化策略和单步结果不同，因此只说明本固定计算预算的耗时关系。

B/C 平均残差块数从约 8,071 降至约 61，标量残差行数保持一致。单线程下，Ceres Summary 的线性阶段由 17.33 ms 降至 1.24 ms，求值由 11.55 ms 降至 6.65 ms，构造由 1.78 ms 降至 1.13 ms；该实验中最大的节省来自 Ceres 对批量问题的线性阶段处理。4 线程进程峰值 RSS 为 A 99.5 MiB、B 97.3 MiB、C 85.2 MiB，包含已加载的全部快照及运行时。

原始结果：`build/abc_benchmark_1step/all.csv`、`summary.json`、`report.md`、`manifest.json`、`validation.log`。清单包含 CPU、亲和性、实际命令、二进制/共享库 SHA-256 和测量代码摘要。

```bash
/usr/bin/python3 scripts/run_abc_benchmark.py \
  --snapshots build/abc_replay_30s/snapshots --output build/abc_1step_repeat \
  --threads 1 4 8 16 --iterations 1 --rounds 5 --max-snapshots 97 --validate 97
```

## 同一冻结问题：八次内部迭代上限

继续使用上述 97 个窗口、4 个优化线程和 5 轮测量；每组 485 个样本，允许各优化器按自身终止条件提前结束。该表与单步表使用同一份初值、关联、先验和权重。

| 后端 | 均值 ms | P95 ms | 平均最终代价 | 峰值 RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| A | 49.77 | 59.63 | 5.149774 | 100.3 |
| B | 67.16 | 69.10 | 5.149811 | 97.5 |
| C | 22.60 | 24.18 | 5.149811 | 86.0 |

B/C 配对时间比中位数为 **2.98**，A/C 为 2.13；B/C 最大相对代价差为 6.86e-12，全部解可用，无显著代价上升样本。批处理的计算收益在多次优化中仍然存在，但 A 实际回放每次只用原生一步，所以这里的 A/C 比值不能替代下面的整帧比较。

结果位于 `build/abc_benchmark_8step/`，具有相同的 CSV、JSON 和 Markdown 报告结构。两次冻结实验总共记录 7,275 个正式样本，预热样本不计入统计。

```bash
/usr/bin/python3 scripts/run_abc_benchmark.py \
  --snapshots build/abc_replay_30s/snapshots --output build/abc_8step_repeat \
  --threads 4 --iterations 8 --rounds 5 --max-snapshots 97 --validate 3
```

本次第二组测量使用 `--validate 0`：97 个窗口的 1/8 次求解已在第一组独立验证进程全部检查，测量进程不重复进行数值检查。

## 30 秒真实回放

正式运行：三组共用配置、4 个优化线程、0.5 倍 bag 播放速度、每帧 3 次外迭代、全部关闭快照写盘。A 每次外迭代保留原生 1 步；B/C 每次最多 8 次内部迭代。每组处理 291 帧 / 873 次优化，scan 时间戳逐帧一致，连续扫描无缺口，全部正常退出且没有空 LiDAR 约束帧。

| 后端 | 每次内部迭代上限 | 整帧均值 ms | 整帧 P95 ms | 最大积压包数 |
| --- | ---: | ---: | ---: | ---: |
| A：native | 1 | 80.65 | 89.97 | 1 |
| B：ceres_scalar | 8 | 261.72 | 276.00 | 68 |
| C：ceres_batch | 8 | 126.90 | 137.77 | 1 |

这轮运行 C/B 的整帧耗时比为 0.485（B/C 约 2.06 倍），A 的整帧耗时仍最低。整帧包含公共最终代价诊断、边缘化与前端，采用不同内部迭代预算；不能将该表解释为三组达到相同真值精度所需时间。播放、等待输入和结束后的 20 秒自动退出等待不计入整帧耗时。

三组首帧优化前的输入摘要、初始代价已经不同。完整回放还存在关联与状态反馈，其输入不是冻结问题。B/C 最终控制点位置互差 RMS 为 0.296 m、最大 1.153 m；这是同一初始坐标系下的原始控制点互差，未做真值对齐，**不是 ATE，也不能证明精度等价**。差异原因未进一步定位，因此当前结论限于计算效率和回放处理情况。

首次把 B/C 都限制为每次 1 次迭代时，两者在约 8 秒后丢失大量 LiDAR 约束；虽然 Ceres 仍返回 usable，后续低耗时属于跟踪失败，不能计入成功 SLAM 的性能结果。该轮原始记录保留在 `build/abc_replay_30s/`，后加的 `tracking_review.json` 将 B/C 判为失败。当时临时增加了 B/C 预算；当前已恢复三组统一单步，并保留连续空 LiDAR 约束帧检查。

该历史回放的原始命令（旧参数，当前入口已移除，不可直接用于新调度）：

```bash
source devel/setup.bash
/usr/bin/python3 scripts/run_abc_replay.py \
  --bag /home/jiadong/workspace/datasets/r3live/degenerate_seq_02.bag \
  --output build/abc_replay_repeat --duration 30 --rate 0.5 \
  --threads 4 --iterations 8 --native-iterations 1 --no-snapshots
```

原始记录：`build/abc_replay_validated/replay_manifest.json`、各组 `optimization.csv` / `frames.csv` / `trajectory/`，以及 `trajectory_differences.json`。这些构建输出未加入版本管理。
