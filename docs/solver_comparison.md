# SLICT A/B/C 对比实现与使用

代码基于 `noetic` 标签 `0a79e9756c532cb2ba93990ff15edfc059367c58`，新增三个可切换后端、共同问题快照、离线计时工具、ROS 回放和数值测试。

[NTU VIRAL eee_03 实测](solver_comparison_eee03.md) 给出当前统一单步调度的完整回放、真值精度与同输入窗口耗时。[历史实测记录](solver_comparison_results.md) 保留旧数值检查及耗时；其中 A=1、B/C=8 回放属于不同迭代预算，不能作为当前整帧基线。

| 组别 / 参数 | 因子构造与求值 | 求解 |
| --- | --- | --- |
| A / `native` | 直接调用原 `Point2PlaneFactorTMN`、`GyroAcceBiasFactorTMN`；逐观测复制支撑，组装全局稠密 Jacobian，再转稀疏 | Eigen SparseLU，原阻尼衰减和步长裁剪 |
| B / `ceres_scalar` | 每个观测注册一个 Ceres 因子，使用与 C 相同的解析内核 | Ceres 2.1，LM，SuiteSparse `SPARSE_NORMAL_CHOLESKY` |
| C / `ceres_batch` | SPLBATCH 按完整参数支撑分组，共享支撑计算，批量注册 LiDAR 和 IMU | 与 B 完全相同 |
| D / `native_batch` | 复用 C 的 evaluator、分组和 batch 求值，通过 NativeRegistry 写入原生 r/J | 与 A 共用矩阵组装、阻尼、裁剪和 SparseLU |

2026-09-07 新增 D，适配范围和 A/D 实测见 [原生后端适配](native_batch_adaptation.md)。Python 工具默认仍运行 A/B/C，指定 `--backends native native_batch` 比较原生适配；独立程序的 `--backend all` 包含四组。

B/C 是隔离批处理收益的主要对照；A/C 比较更换构造、求值和求解方式的总影响。相同迭代上限不代表相同收敛程度，必须同时看最终代价。A 是**统一模型后的原生求解流程**：原 Noetic 的先验与 Ceres 路径不等价，因此本实验给三组采用共同的平方根边缘化先验和初始锚定。A/C 结果不能直接当作未经修改的 Noetic 程序的加速比。

未设置 `/solver_backend` 时仍走原来的 `/use_ceres` 选择。比较模式按四阶均匀 SO(3) + R³ 样条实现，SO(3) 采用右乘更新。每条 IMU 保留原生 12 维残差，参考 bias 在窗口第一次外迭代冻结。LiDAR 保留每点时间、点面关联、法向和权重。

## 编译和数值验证

依赖位置及 Ceres 2.1 / SuiteSparse 7 的本地兼容处理见 [Noetic 构建记录](build_noetic.md)。已有依赖目录时：

```bash
cmake --build build/noetic --parallel 16
ctest --test-dir build/noetic --output-on-failure -R slict_solver_comparison
source devel/setup.bash
```

Zsh 使用 `source devel/setup.zsh`。构建并行度与优化线程数分别控制。

`slict_solver_comparison_test` 覆盖原生因子与 B/C 的残差、解析 Jacobian、正规方程、右乘有限差分、零/微小/一般旋转、区间端点、部分/空 Jacobian 请求、冻结变量、非零先验偏移、四元数符号、1/4 线程、1/8 次求解、快照回读及损坏检测。边缘化先验另外与独立 QR 消元后的二次代价变化比较，检查 Hessian 与梯度符号。

2026-09-06 验证：`--parallel 16` 编译及上述 CTest 通过；NTU VIRAL/R3LIVE 的六组 launch 参数解析通过；R3LIVE 前 5 秒调度检查通过，调用方覆盖 `comparison_iterations=8` 时启动即被拒绝。随后完成 NTU VIRAL `eee_03` 全序列：三组各处理 1,806 帧、5,418 次求解调用，每帧求解/去畸变/重新关联计数均为 3/3/3，scan 时间戳完全一致，全部正常退出且没有空 LiDAR 约束帧。记录在 `build/abc_eee03_single_step/replay_manifest.json`；真值精度和独立冻结问题计时见 [本次报告](solver_comparison_eee03.md)。数值测试还覆盖连续单步调用的信赖域半径传递。

## 先运行无 ROS 的合成问题

```bash
./devel/lib/slict/slict_solver_benchmark \
  --synthetic 3 --knots 20 --lidar 1000 --imu 60 \
  --backend all --threads 1 --iterations 1 \
  --validate 3 --warmup 1 --repeat 5 --output build/abc_synthetic.csv
```

`--generate build/abc_synthetic_windows` 可保存合成快照。所有后端每次从同一初值重置，输入摘要随 CSV 输出。`--validate N` 对前 N 个窗口比较 H、梯度和归一化代价，超过 `1e-9` 即退出失败；还比较 B/C 在 1/8 次迭代后的完整状态，最大向量差超过 `1e-6` 即失败。

## 真实 bag 回放和快照采集

三组统一执行以下循环，默认每帧三轮：

```text
当前关联 → 优化一步 → 写回状态、IMU 传播 → 去畸变 → associate → 下一轮
```

每次优化固定 `comparison_iterations=1`，由同一段 Estimator 前端完成更新、去畸变和重新关联。`--outer-iterations`（launch 中为 `outer_iterations`）只调整三组共同的循环轮数。旧的 `--iterations`、`--native-iterations` 和 launch 的 `iterations` 参数已移除；即使其他入口将 ROS 参数 `/comparison_iterations` 改成大于 1，比较后端也会拒绝启动。离线冻结问题仍可运行多步实验。

一次 Ceres 调用最多进行一次 LM 迭代；若该步被拒绝或已满足收敛条件，写回的状态可能不变，仍进入同一去畸变/关联流程。单步不表示残差只求值一次，也不表示三种求解器得到相同增量。

B/C 的连续单步调用接续 Ceres 返回的信赖域半径，跨外迭代及窗口传递；第一步从 Ceres 默认的 `1e4` 开始。此前每次重新建 Problem 并调用 Solve 都把半径重置为 `1e4`，会丢失已调整的阻尼。在 `eee_03` 的单步试验中 B 因此逐步失去 LiDAR 约束；接续半径后仍保持每轮一步，额外在 CSV 中记录 `ceres_initial_trust_region_radius` 和 `ceres_final_trust_region_radius`，回放脚本检查相邻调用的连续性。

共同设置在 [abc_common.yaml](../config/abc_common.yaml)。`reassociate_steps=2` 表示每轮重新去畸变、关联窗口内最新两个点云，首帧处理整个三帧窗口；它不是每隔两步才关联。最后一轮之后同样执行这套前端更新。`reassoc_rate=0` 仅关闭因子求值内部的关联。

优先用论文的 **NTU VIRAL** 测试，下载入口和序列选择见 [论文数据集说明](slict_datasets.md)。[统一 launch](../launch/run_abc.launch) 根据 `dataset` 加载传感器配置：

- `ntuviral`：[配置](../config/abc_ntuviral.yaml)，使用水平 16 线雷达 `/os1_cloud_node1/points` 和 `/imu/imu`。保留原 NTU 外参及扫描结束时间戳约定，与 SLICT2 论文的传感器选择一致。
- `r3live`：[配置](../config/abc_r3live.yaml)，使用已有 Livox→Ouster 转换器、每点 ns offset、header 时间基准、m/s² 加速度。外参来自本机 ClampSLAM 的 `config/r3live/lidar.yaml`。

下载并解压 NTU VIRAL bag 后执行（替换实际路径）：

```bash
source devel/setup.bash
/usr/bin/python3 scripts/run_abc_replay.py \
  --dataset ntuviral --bag /path/to/eee_03.bag \
  --output build/abc_ntuviral_capture \
  --duration 30 --rate 0.5 --threads 4 --outer-iterations 3 \
  --snapshot-stride 3 --snapshot-limit 200
```

输出目录必须是新目录。工具顺序启动 A/B/C，各自使用独立 ROS master、日志、轨迹和 CSV。`--dataset` 必须显式指定，避免误用外参。`--duration 0` 使用整个 bag；`--timeout` 是每个后端允许的墙钟秒数，全序列测试可设为 1800。工具检查节点异常、不可用解、处理时段、相同 scan 时间戳、连续扫描间隔和帧数；默认连续超过 5 帧没有 LiDAR 约束也判失败。

此外，回放工具逐帧检查优化记录的轮次顺序、内部迭代上限为 1，以及实际优化调用、后处理去畸变、重新关联的次数均等于共同外迭代数。结果写入 `single_step_association_cadence`；缺失或不一致会退出失败。

A 首次外迭代的输入按 stride 导出；下一帧的共同先验在上一帧最后一次求解后的状态上生成。退化数据上的历史单步跟踪失败仍需如实计入失败率，不通过单独增加 B/C 内部迭代数改变本实验口径。

上述采集运行只有 A 写出快照，其整帧耗时包含额外采集开销。正式对比整帧性能时，在新输出目录重跑并添加 `--no-snapshots`，让三组都关闭快照写盘；冻结问题的 `total_ms` 始终不含该开销。

直接切换单组：

```bash
roslaunch slict run_abc.launch dataset:=ntuviral \
  bag_file:=/path/to/eee_03.bag \
  backend:=ceres_batch threads:=4 outer_iterations:=3 \
  output_dir:=$PWD/build/abc_live_c duration:=30
```

比较模式显式拒绝非四阶样条、求值内重关联、鲁棒核、pose/velocity propagation 因子、外迭代代价提前终止、非单步优化、零次外迭代/关联和实时预算截断。当前实验关闭回环、RViz 和 spline fitting。遇到后端失败会终止，不自动换后端。

回放输出：

- `snapshots/*.slict`：A 的冻结窗口，包含观测、完整初值、参考 bias、固定变量、先验、阻尼和时间基准。二进制版本 1，显式小端编码；FNV-1a 校验用于发现意外损坏，不是安全哈希。
- `<backend>/optimization.csv`：每次外迭代的输入摘要、观测数、控制点数、残差块数、各阶段计时、接受/拒绝步数、最终代价、状态是否可用。
- `<backend>/frames.csv`：完整帧处理时间、scan 时间戳、处理后积压包数，以及 `optimizer_calls`、`post_opt_deskew_passes`、`post_opt_association_passes`；不含等待输入和本 CSV 写盘时间，包含快照导出与公共诊断开销。次数不含每帧首次优化前的初始化关联。
- `<backend>/trajectory/spline_log.csv`、`KfCloudPose.pcd`：轨迹与控制点；没有真值时不能将三组互差称为 ATE。
- `replay_manifest.json`：数据集、单步及共同轮数、调度次数检查、实际命令、帧数、时段、整帧均值/P95、退出检查。

回放中同时修复了原有退出问题：Estimator 超时通过 ROS shutdown 退出，主线程等待数据及地图工作线程后保存轨迹；SensorSync 两个循环检查 `ros::ok()` 并在析构前 join，避免 joinable thread 导致 SIGABRT。

## 冻结同一批真实问题做正式计时

回放结束并停止其他计算负载后运行：

```bash
/usr/bin/python3 scripts/run_abc_benchmark.py \
  --snapshots build/abc_ntuviral_capture/snapshots --output build/abc_benchmark \
  --threads 1 4 8 16 --iterations 1 --rounds 5 \
  --max-snapshots 200 --validate 3
```

工具先在单独进程进行数值检查，再为每个后端、线程数和重复轮次启动独立进程；每个进程预热整批输入一次，正式运行一次。轮换后端顺序，三组匹配快照摘要后才出报告。BLAS/MKL 线程固定为 1，关闭 OpenMP 动态和嵌套并行，记录共同 CPU 亲和集合。这里的 `threads` 只控制优化器；SLICT 前端仍使用原线程策略。

离线比较多次优化的代价与时间可加 `--iterations 1 5`；快照实验固定关联，内部没有前端去畸变或重新关联，不能替代主回放。每个输出目录包含 `all.csv`、`summary.json`、`report.md`、实际命令和二进制 SHA-256。汇总包含均值、P50/P95/P99、配对 A/C 和 B/C 时间比、B/C 最大相对代价差和进程峰值 RSS。RSS 包含进程运行时、载入的全部快照及优化内存；正式测量进程不执行 H/J 数值检查。

离线的 `--ceres-initial-radius` 显式指定 B/C 在每个冻结问题上的共同初始信赖域半径（默认 `1e4`），验证进程与计时进程使用同一值。离线问题彼此独立，不把上一个快照的半径传到下一个快照。比较实际连续求解的稳定阶段时，应同时记录回放中的半径分布并选择相应设置。

### 计时口径

`total_ms` 是状态重置、问题构造、求解与问题析构的墙钟总时间。A 的稠密清零和逐因子构造落在求值阶段，B/C 的 `Commit()` 落在构造阶段。快照读取、校验和独立最终代价检查不计入该值；公共最终代价检查另列 `diagnostic_ms`。

`solve_ms` 已包含其内部 `evaluate_ms`、`linear_ms` 等分项，不能重复相加。A 可直接细分 `sparseView`/正规方程和 SparseLU；Ceres 的细分取其 Summary，不能把无法取得的项目填成另一个阶段的耗时。

B/C 的 `successful_steps` / `unsuccessful_steps` 保留 Ceres Summary 原值，其中 successful 包含初始评估；`iterations` 已排除初始评估。比较迭代预算请使用 `iteration_limit` 和 `iterations`。

`marginalize_ms` 为共同边缘化时间，仅每帧最后一次外迭代非零；`input_ms` 包括 ROS 输入准备和结果写回。`backend_with_diagnostics_ms` 是 core、输入/写回、诊断和边缘化之和，不包括快照 hash、写盘或整个前端。评估实际帧预算用 `frames.csv`。

### 代码入口

- [统一接口](../include/slict/solver_comparison.h) 与 [快照格式](../src/comparison/snapshot_io.cpp)。
- [原生求解](../src/comparison/native.cpp)、[B/C 构造](../src/comparison/ceres_backend.cpp)、[共享解析内核](../src/comparison/evaluators.h)。
- [共同先验及边缘化](../src/comparison/problem.cpp)、[ROS 窗口桥接](../src/comparison/ros_bridge.cpp)。
- [数值测试](../tests/solver_comparison_test.cpp)、[离线脚本](../scripts/run_abc_benchmark.py)、[回放脚本](../scripts/run_abc_replay.py)。

SPLBATCH 基础快照来自本机 ClampSLAM 的 `third_party/splbatch`，固定提交 `e2892937c356b34d2873c91cea9b044a6c9d897a` / 0.1.1。2026-09-07 增加面向 D 的注册出口补丁，evaluator、分组算法和 batch 求值保持不变。[来源记录](../third_party/splbatch/VENDORED.md) 和许可证随代码保留。当前未启用 Hessian 压缩、直接 H/g 装配或增量 Problem 管理。
