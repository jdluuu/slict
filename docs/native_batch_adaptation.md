# SPLBATCH 接入 SLICT 原生后端

2026-09-07，新增 **D / `native_batch`**。它消费与 C 完全相同的 `SegmentBatchCostFunction`，使用 A 的矩阵装配、阻尼、SparseLU 和增量裁剪。原 A / `native` 保留为对照；本次没有加入直接 H/g 累加、正规方程压缩或增量 Problem 管理。

后续已加入 [P1 支撑查找缓存](splbatch_p1_results.md)，修改普通 channel 的构造流程并在 `batch_binding.hpp` 增加辅助函数。下文的源码一致性说明与测量描述原生适配完成时的版本；P1 的变化和新测量单独记录。

这里的 A 是此前 A/B/C 对比中已统一残差、权重和边缘化先验的原生后端，保留 SLICT 原生求解流程；它不是未经修改的 noetic tag。A/D 只对比本次批处理适配的增量影响。

## 实现范围

```text
同一套 LidarEvaluator / ImuEvaluator
  → 同一套 AddObservation、分组与 PrepareObservation
  → 同一个 SegmentBatchCostFunction
  → ResidualRegistry
      ├─ CeresResidualRegistry → Ceres Problem → Ceres LM
      └─ NativeRegistry       → 全局 r/J → A 的 SparseLU 求解流程
```

[residual_registry.hpp](../third_party/splbatch/include/splbatch/residual_registry.hpp) 新增注册接口，接收批次的 `unique_ptr<ceres::CostFunction>` 和非拥有的参数指针。[problem_batcher.hpp](../third_party/splbatch/include/splbatch/problem_batcher.hpp) 只调整注册出口及其连接：分组算法、待提交观测、预计算和 batch 构造保持原实现。现有 `ProblemBatcher(ceres::Problem&)` 和 channel 的 `CommitTo(ceres::Problem&)` 继续可用；自定义 registry 的 batcher 拒绝调用 Ceres `Solve()`。

当前接口保留 `ceres::CostFunction` 类型依赖，D 不构造 `ceres::Problem`，也不调用 Ceres 优化器。SLICT 的 evaluator 和 `segment_batch_cost_function.hpp`、`batch_binding.hpp`、Jacobian writer 等计算头文件的 SHA-256 与适配前相同。固定上游版本及本地补丁范围见 [来源记录](../third_party/splbatch/VENDORED.md)。

[native_batch.cpp](../src/comparison/native_batch.cpp) 实现 NativeRegistry：

1. 为当前状态注册参数指针到全局列的映射。旋转和位置列分别为 `6*i` / `6*i+3`，bias 位于最后六列。
2. 调用与 C 相同的注册、观测添加、`Commit()`，持有生成的 batch 对象；为每个 batch 分配可复用的行主序 Jacobian 缓冲区。
3. 每次线性化按 batch 并行调用现有 `Evaluate()`，把残差放到该批的独立行范围，再把局部 Jacobian 写入列主序的全局 J。
4. 固定控制点保留参数数值，对应 Jacobian 请求设为空，全局导数列置零。旋转沿用当前 SLICT 的 `[J_SO3, 0]` 解析约定，仅映射前三列到局部增量，避免重复应用四元数 Plus Jacobian。
5. 先验继续由共同 `LinearizePrior()` 添加一次。求值失败在退出并行区域后报告。

状态存储必须比 registry 活得更久，注册后不能移动控制点存储。每次 Solve 从输入状态复制后构建 registry，在状态对象销毁前释放批次和缓冲区。多次内部迭代复用注册和缓冲区，但每次 batch `Evaluate()` 都根据更新后的状态重新计算 runtime；ROS 调度每次调用仍最多一步。

[native.cpp](../src/comparison/native.cpp) 中 A/D 共用同一段求解代码，只选择不同的 r/J 求值入口。两者均执行全局 `J.sparseView()`、`H=JᵀJ`、`rhs=-Jᵀr`，相同标量阻尼、SparseLU 分析与分解、整体增量范数裁剪和右乘旋转更新。没有因子级接受/拒绝逻辑变化，首帧单步代价上升等原生行为也应一致。

新增 `UsesCeresSolver()` 显式区分 A/D 与 B/C，避免把 D 错当 Ceres 后端派发，或错误记录/检查信赖域半径。ROS 前端仍共用“优化一步 → 状态更新、IMU 传播 → 去畸变 → associate”，每帧三轮，最后一轮执行共同边缘化。

## 数值与接口检查

使用 `cmake --build build/noetic --parallel 16` 构建，`slict_solver_comparison` CTest 通过。新增测试包括：

- 与原生 r/J 的逐项比较，非连续固定控制点、带偏移先验、四元数符号变化、仅 LiDAR / 仅 IMU / 仅先验 / 空问题。
- 同一 registry 在状态更新后重复求值，以及残差求值与 Jacobian 求值交替调用。
- 1/4 线程、1/8 步、裁剪阈值 `0.0001` / `0.5` / 无裁剪的 A/D 最终状态与代价比较。
- 观测顺序打乱时 H/g 不变，批次提交幂等、提交后拒绝添加观测、原 Ceres channel 接口兼容。
- 原有 B/C 残差、Jacobian、有限差分、边缘化、求解和快照检查仍通过。

原生适配器按 batch 顺序返回残差行；相同支撑的观测若不连续，行顺序会变化，但 H/g 与目标函数不变。实际比较输入统一按 span 排序，A/D 的行顺序也一致。

## eee_03 实测

使用前次采集的 101 个相同窗口，各线程配置和后端独立进程测 5 轮，共 4,040 个正式样本。A 和 D 均在本次编译后重新计时，未拿历史 A 的耗时作分母。CPU 为 AMD Ryzen 9 9950X，亲和集合 0–31；BLAS/MKL 单线程。测试时没有同时编译或运行 ROS 回放。

| 优化线程 | A 原生均值 / P95 ms | D 适配均值 / P95 ms | D 均值变化 | A/D 配对时间比中位数 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.14 / 22.69 | 9.90 / 12.08 | 降低 48.3% | 1.94 |
| 4 | 9.62 / 12.58 | 8.54 / 10.54 | 降低 11.3% | 1.12 |
| 8 | 8.19 / 10.98 | 7.95 / 10.36 | 降低 2.9% | 1.02 |
| 16 | 7.42 / 10.44 | 8.06 / 10.69 | 增加 8.7% | 0.90 |

4 线程平均分项如下。求值包括局部结果到全局 J 的写入，装配仍为全局 J 转稀疏及正规方程计算；D 构造包含完整分组、批次注册和局部缓冲区分配。

| 阶段 | A ms | D ms |
| --- | ---: | ---: |
| 构造 | 0.002 | 1.321 |
| 求值 | 4.527 | 2.042 |
| 矩阵装配 | 4.158 | 4.120 |
| 核心总时间 | 9.624 | 8.539 |

批量求值显著减少重复支撑计算，但矩阵装配保持原流程，且多出约 1.3 ms 的构造开销。随着原生逐观测求值的线程数增加，批处理节省的求值时间逐渐小于新增开销，16 线程下 D 反而更慢。这个最小适配不能视作对所有线程配置的普遍加速。直接按批累加 H/g 可另作后续实验，本次没有把它混入比较。

4 线程进程峰值 RSS 为 A 123.1 MiB、D 131.4 MiB；RSS 包含载入全部快照及运行时。D 保留全局 J 的同时增加了局部 Jacobian 缓冲区，因此本实现没有取得内存节省。

101 个窗口的 A/D 最大 H 相对差 `1.41e-15`、梯度相对差 `2.34e-12`、初始代价相对差 `4.16e-13`；1/8 步验证的最大状态差 `2.87e-11`，最终代价相对差 `1.02e-10`。正式单步计时最终代价最大相对差 `2.96e-12`，所有解可用。两组均保留首个窗口的代价上升行为，每个线程配置均为 5 个重复样本；没有剔除窗口以改变性能或代价结论。

完整回放使用同一 eee_03 bag 的水平 LiDAR `/os1_cloud_node1/points` 和 `/imu/imu`，4 线程、1 倍速、每帧三次单步优化，关闭快照采集。按 D → A 顺序各回放一次：两者均处理 1806 帧、5418 次优化，每次恰好一步；扫描时间戳一致，每帧均有三次优化后去畸变和 associate，所有解有效，没有空 LiDAR 约束帧。

| 完整回放指标 | A 原生 | D 适配 |
| --- | ---: | ---: |
| 单次优化核心均值 / P95 ms | 9.714 / 12.967 | 9.215 / 12.234 |
| 含公共诊断、边缘化及输入/回写的后端均值 ms/次 | 21.375 | 20.897 |
| 整帧均值 / P95 ms | 92.005 / 108.360 | 91.038 / 106.111 |
| 最大队列包数 | 11 | 3 |
| 棱镜位置 ATE RMSE cm | 2.622 | 2.598 |
| 位置误差 P95 cm | 4.726 | 4.705 |

本次完整回放的核心优化均值降低 **5.1%**，整帧均值仅降低 **1.1%**。回放时 D 构造均值为 1.876 ms，高于冻结窗口的 1.321 ms；求值仍从 A 的 4.587 降到 D 的 2.067 ms，但原生装配仍约 4.16 ms。共同的最终代价复核约 10.0 ms/次，加上边缘化和前端处理，使整帧收益明显小于求值收益。两次回放均完整排空队列；单次队列峰值和约 1% 的整帧差异不应视作稳定吞吐提升，需要重复回放才能判断其波动。

精度在全部 2990 个真值时间点上评估，覆盖约 150.75 s；真值本身晚于 bag 开始约 28.71 s。直接在 GT header 时间上求值 SO(3)+R3 样条，使用数据集棱镜外参，然后仅作刚体位置对齐，不拟合尺度或时间偏移。A/D 精度基本相当；单次回放的微小 ATE 差异不代表适配提升了算法精度。在线关联会随轨迹更新，两组回放不是冻结的同一组约束，数值等价性的主要证据来自前面的相同窗口验证。

结果与审计材料：

- [持久化完整结果及 SHA-256](native_batch_results.json)：环境、输入、冻结窗口各阶段统计、完整回放、精度和代码一致性记录。
- [冻结窗口原始 CSV](../build/native_batch_eee03_frozen/all.csv)、[运行清单](../build/native_batch_eee03_frozen/manifest.json)、[数值验证日志](../build/native_batch_eee03_frozen/validation.log)。
- [完整回放清单](../build/native_batch_eee03_replay/replay_manifest.json)、[精度 JSON](../build/native_batch_eee03_accuracy/accuracy.json)、[轨迹与误差图](../build/native_batch_eee03_accuracy/trajectories.png)。

## 复现

编译、数值检查与冻结窗口计时：

```bash
cmake --build build/noetic --parallel 16
ctest --test-dir build/noetic --output-on-failure -R slict_solver_comparison
/usr/bin/python3 scripts/run_abc_benchmark.py \
  --snapshots build/abc_eee03_capture/snapshots \
  --output build/native_batch_eee03_frozen \
  --backends native native_batch --threads 1 4 8 16 \
  --iterations 1 --rounds 5 --max-snapshots 101 --validate 101
```

上述快照已由此前的 A 全序列回放采集，保留相同初值、关联、权重、先验和阻尼。独立验证进程额外检查 A/D 的 1/8 步状态一致性；正式计时使用 1 步。每轮全批预热一次，后端顺序轮换。计时包含注册、缓冲区分配、求值/装配/求解和析构，不包含快照 I/O、公共最终代价复核和边缘化。

完整回放与棱镜真值精度：

```bash
source devel/setup.bash  # zsh 使用 devel/setup.zsh
/usr/bin/python3 scripts/run_abc_replay.py \
  --dataset ntuviral \
  --bag /home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag \
  --output build/native_batch_eee03_replay --backends native_batch native \
  --duration 0 --rate 1 --threads 4 --outer-iterations 3 \
  --timeout 600 --no-snapshots
/usr/bin/python3 scripts/evaluate_ntuviral.py \
  --ground-truth build/abc_eee03_inputs/ground_truth.csv \
  --prism-calibration /home/jiadong/workspace/datasets/ntu/eee_03/leica_prism.yaml \
  --replay build/native_batch_eee03_replay --backends native native_batch \
  --output build/native_batch_eee03_accuracy
```

输出目录必须是新目录，复跑时更换上述目录名。回放包含全部公共诊断、边缘化及前端处理，用 `frames.csv` 衡量整帧时间。单次优化加速比不能直接当成整帧加速比。

独立基准程序的 `--backend all` 现在包含 A/B/C/D；Python 回放和计时脚本默认仍为 A/B/C，可用 `--backends native native_batch` 只测 A/D。NTU 评估脚本支持同样的后端选择，并在同一批真值时间戳上评估。
