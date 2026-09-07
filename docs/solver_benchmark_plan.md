# SLICT 自写求解器与 SPLBATCH + Ceres 效率对比方案

基于 2026-09-06 检查的本地代码。A/B/C 后端、共同快照、数值测试和回放工具现已实现，使用方法和实际限制见 [实现文档](solver_comparison.md)。下文保留原始实验设计，拟议扩展不代表全部已经实现。

当前主回放固定 A/B/C 每轮一步优化，随后更新状态、去畸变和重新关联；每帧共同执行三轮，可通过 `--outer-iterations` 同时调整。冻结问题的多步实验单独记录。优先使用论文的 NTU VIRAL 数据，[序列和下载入口](slict_datasets.md) 已整理。

前置验证已完成：SLICT 全部 12 个目标编译通过；同一套 Ceres 2.1 下，SPLBATCH 的 3 个测试及拟合示例通过。环境、依赖处理和复现命令见 [构建记录](build_noetic.md)。

## 1. 比较对象与目标

| 组别 | 因子构造和求值 | 优化器 | 用途 |
| --- | --- | --- | --- |
| A | SLICT 原生逐因子解析计算，组装全局 Jacobian | `tmnSolver`：Eigen SparseLU、原阻尼和步长限制 | 衡量替换现有方案的总收益 |
| B | 相同数学模型，逐观测注册 Ceres 因子 | Ceres | 隔离批处理的收益 |
| C | 相同数学模型，通过 SPLBATCH 按参数支撑分组、缓存和批量求值 | 与 B 完全相同的 Ceres 配置 | 用户希望引入的方案 |
| D（可选） | 复用 C 的批量计算内核，输出原生局部 Jacobian | 与 A 相同的组装和求解流程 | 区分计算复用、问题组装与求解器的影响 |

核心结论分别报告 **A/C 的整体替换收益** 和 **B/C 的批处理收益**。D 是归因实验；第一阶段不同时引入直接累积 Hessian、压缩因子或增量问题管理。

代码版本：

- SLICT：`noetic`，`0a79e9756c532cb2ba93990ff15edfc059367c58`。
- 用户所说的 `workspace/clampslam` 对应 `/home/jiadong/workspace/ClampSLAM`。
- 其中 `ct_batch` 已整理为 `src/ClampedSLAM/third_party/splbatch`，本次检查提交 `e2892937c356b34d2873c91cea9b044a6c9d897a`，项目版本 0.1.1。
- SPLBATCH 是批处理和 Ceres 注册库，样条模型、观测绑定及解析 Jacobian 由应用提供。不能把整个 ClampSLAM 算法当作 C 组，否则同时改变了前端和轨迹模型。

## 2. 当前代码揭示的比较陷阱

入口是 [Estimator.cpp](../src/Estimator.cpp) 的 `LIOOptimization()`；`use_ceres=0` 调用 [tmnSolver.cpp](../src/tmnSolver.cpp) 的 `Solve()`，另一条路径构造 `ceres::Problem`。

目前两条路径并非只差一个求解器：

1. `tmnSolver` 每次调用只计算一次更新，使用 `H + lambda / 2^k * I`、`SparseLU` 和 `dx_thres` 裁剪；Ceres 每次调用可执行多轮 LM，还有接受/拒绝步骤。不能拿“各一次 Solve”或相同迭代数字认定计算量相同。
2. 原生路径支持 `fuse_marg`，默认 NTU 配置开启；当前 Ceres 的 `LIOOptimization()` 没有注册同一份边缘化先验，却会按 `start_fix_span` 固定开头控制点。必须统一先验与固定变量，再解释求解器速度。
3. Ceres LiDAR 的关闭鲁棒核条件是 `lidar_loss_thres == -1`，而 NTU 配置写的是 `-10.0`，仍会构造 CauchyLoss；原生 LiDAR 求值没有对应鲁棒处理。第一阶段显式使用 `-1`，后续有鲁棒核的实验须让三组采用相同的逐观测损失。
4. 两条路径的 bias 先验取值时机不同：原生保留本窗口第一次外迭代时的 bias，Ceres 在创建因子时使用当时的 bias。需要把参考 bias 放入共同输入快照。
5. Ceres 点面因子可在 `Evaluate()` 内重关联；冻结问题实验关闭它，固定 `featureSelected`、`imuSelected`、点面关联、法向和权重。完整回放再按共同调度更新关联。
6. [TimeLog.msg](../msg/TimeLog.msg) 中同名计时并不等价：原生 `t_compute` 包含稠密转稀疏、正规方程、分解、求解和状态更新；Ceres 的 `t_compute` 是整个 `ceres::Solve()`，已经包含残差/Jacobian 求值。不能直接逐列相减。
7. 原生求解器含文件级维度变量、函数静态 bias 和窗口状态。离线回放优先使用独立进程，或把这些状态收进后端实例，避免比较模式互相污染。

保留原配置回放作为 A 的实际使用基线，但明确区分“原配置表现”和“同一问题的控制实验”。如果规范化模型需要修复原先实现中的问题，单独记录修复前后，不把修复收益混进批处理收益。

## 3. C 组的最小接入范围

在 `FactorSelection()` 之后、`LIOOptimization()` 修改状态之前导出 `OptimizationSnapshot`，内容包含控制点、时间基准、样条阶数与间隔、观测及其支撑、bias/参考 bias、噪声权重、固定变量、先验矩阵和线性化点。对输入、观测集合与配置保存校验值。

统一后端入口为 `Solve(snapshot, options) -> state + metrics`，现已支持 `native`、`ceres_scalar`、`ceres_batch`。

- 保留 SLICT 的均匀 SO(3) + R³ B-spline 和右乘更新 `R <- R Exp(dtheta)`；先支持当前配置的 `SPLINE_N=4`，其他阶数必须显式检查或实现，不能静默按四阶处理。
- 点面因子以 [Point2PlaneFactorTMN.hpp](../include/factor/Point2PlaneFactorTMN.hpp) 和 [PointToPlaneAnalyticFactor.hpp](../include/factor/PointToPlaneAnalyticFactor.hpp) 为数学依据。按完整、有序的旋转/位置参数支撑分组，每个点保留自己的时间戳、点、平面及权重。
- `PrepareObservationImpl()` 缓存固定时间系数；`PrepareBatchImpl()` 每次求值重新计算同一支撑共享的相邻旋转 log 和逆右 Jacobian。不同时间的完整位姿不能简单共用，也不能跨迭代缓存依赖控制点数值的结果。
- IMU 以 [GyroAcceBiasFactorTMN.hpp](../include/factor/GyroAcceBiasFactorTMN.hpp) 为依据，保留每个观测的 **12 维**残差、共同 bias 参数块、参考 bias 和时间导数缩放。ClampSLAM 中现成的 IMU/bias 拆分不能直接代替 SLICT 模型。
- 第一阶段先处理 LiDAR，再处理 IMU；先验通常作为一个整体稠密因子。优先使用普通 `ProblemBatcher`，将 `Commit()` 计入构造耗时。
- SLICT 解析旋转因子配合专用的 4×3 local parameterization，第四列填零、前三列承载局部导数。复用这一约定并检查 Ceres 链式乘积；不能只换成标准 QuaternionManifold。
- 同一先验必须对照原生实际使用的 `H,b`、状态差和线性化点验证。不能仅凭 `HbToJr()` 的名字认定平方根化、符号或负特征值截断完全等价。若无法表达为相同的最小二乘模型，先做关闭边缘化且统一锚定的控制实验，再单独处理先验差异。
- 正式接入时固定 SPLBATCH 的提交，并使三个后端链接同一版本 Ceres。不要把现有 ClampSLAM 的 Ceres 2.2 二进制与 SLICT 的 Ceres 2.1 混用。

## 4. 实验按三个层次推进

**第一层：数值一致性。** 使用相同快照，逐类比较残差、局部 Jacobian、归一化代价、`Jᵀr`、`JᵀJ`。原生代价是残差平方和，Ceres 报告含 `1/2`，需要统一口径。建议从残差/Jacobian 相对与绝对混合容差 `1e-10`、正规方程 `1e-9` 开始；同时用右乘扰动有限差分校验导数。覆盖零/小/一般旋转、区间边界、部分和空 Jacobian 请求、冻结变量、bias、先验和离群点。

B/C 使用完全相同的 Ceres 选项，对照一轮和多轮求解结果。A/C 若要对比单步增量，必须先统一阻尼、变量顺序及裁剪策略；各自原生优化策略下，比较达到相同最终代价/精度所需时间，不要求迭代轨迹一致。

**第二层：冻结真实问题的离线计时。** 从基线回放采集约 200–500 个窗口，覆盖不同点数、窗口位置和退化程度。快照读取不计入优化耗时，重置输入单独计时，禁止后端改变下一组输入。先预热，再按交错顺序重复至少 5 轮，运行不同后端时不同时运行其他计算任务。

| 维度 | 建议取值 |
| --- | --- |
| LiDAR 观测数 | 1k、4k、8k；16k 仅作为扩展压力实验 |
| 控制点数 | 记录真实分布；扩展测试 20、40、80 |
| 求值线程 | 1、4、8、16，三组使用相同 CPU 集合 |
| 批处理范围 | LiDAR；LiDAR + IMU |
| 关联策略 | 冻结；之后再测共同外迭代重关联 |
| 优化终止 | 固定计算预算；达到共同目标代价，两类分别报告 |

批次过大可能减少 Ceres 可调度的并行任务；记录批次数和每批观测分布，必要时增加明确的分块粒度实验。先测默认分组，再调参，保留未调参结果。

**第三层：同一 SLICT 前端的完整回放。** 三组共用传感器处理、采样、地图、外参和配置。先关闭回环、RViz、实时预算截断和额外代价打印，检查每组处理相同输入时段、没有静默丢帧；离线顺序处理用于算力比较，实时 1× 回放另测超时和积压。轨迹反馈导致关联变化是端到端实验的一部分，因此不能用它替代冻结问题实验。

本机已有 `/home/jiadong/workspace/datasets/r3live/degenerate_seq_00.bag`、`01`、`02` 和 `hku_campus_seq_00.bag`，可覆盖退化与常规场景，但本次未验证它们对 SLICT 的话题、外参和时间单位适配。也可使用已有 launch 支持的 NTU VIRAL 数据。没有真值时仅报告轨迹重复性和互差，不称作 ATE；有真值后统一对齐、采样和评估 ATE/RPE。

## 5. 指标及计时边界

按单次外迭代和完整帧分别记录：

- 构造：状态/参数注册、系数准备、分组、分配和 `Commit()`。
- 求值：仅残差、残差及 Jacobian、先验求值；原生另外记录稠密清零和 `sparseView()`。
- 线性阶段：`JᵀJ/Jᵀr`、排序/符号分析、数值分解、回代；无法独立获取的 Ceres 阶段保留为其公开汇总，不伪造细分。
- 优化总耗时：从开始构造到状态写回，单列边缘化与析构；另计包含这些阶段的完整后端总耗时。
- 系统指标：完整帧 wall time、P50/P95/P99、峰值 RSS、吞吐、超时/失败/丢帧数量；记录观测数、参数数、批次数、求值次数、接受/拒绝步骤和最终代价。

Ceres 的 residual/Jacobian/linear-solver 分项属于 `Solve()` 内部，不能再次与 `Solve()` 相加。批次鲁棒化的辅助残差行也不能算成额外观测。

使用 `steady_clock`，CSV 写出放在测量区间外。统一编译器、Release 优化选项、Ceres/Eigen/SuiteSparse、CPU 亲和性和后台负载。当前 `MAX_THREADS` 定义在 [utility.h](../include/utility.h)，需要增加统一线程选项才能扫描线程数；**构建 `-j16` 与运行时线程数是两回事**。控制 BLAS 线程和嵌套并行，记录实际生效值。

最终报告每个序列的逐轮结果和配对加速比，不能只汇总最快一次或混合不同观测规模。SPLBATCH 在 ClampSLAM 的历史收益只能作为动机，不能直接当作 SLICT 的预期百分比。

## 6. 实施顺序与决策

1. 加共同快照导出和分阶段计时，记录原配置 A 基线。
2. 统一 B 的观测、bias 参考、固定变量、损失和先验；完成 A/B 数值检查。
3. 接入 C 的 LiDAR，再接入 IMU；完成 B/C 数值与求解一致性检查。
4. 完成冻结快照的观测数/控制点数/线程数实验，明确耗时主要在哪个阶段。
5. 做完整回放及精度评估；必要时加 D 区分缓存与求解器贡献。

建议预先设定工程验收门槛，例如在代表性序列上，精度与失败率满足同一要求，后端均值至少减少 20%，P95 不退化，并报告是否达到帧预算。20% 是建议门槛，不是预测结果。如果只有 Jacobian 加快而完整后端没有收益，应保留原生求解器并考虑单独复用批量求值。

预期交付物：可切换的同模型后端、数值测试、快照回放工具、逐帧 CSV、机器/版本清单、性能与精度报告。批处理 Hessian 压缩和增量复用放在这个结论之后分别评估。
