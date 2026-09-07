# NTU VIRAL eee_03：统一单步 A/B/C 实测

2026-09-06，完整回放 `eee_03` 后，C 的整帧平均耗时为 **70.31 ms**，比 B 低 **49.5%**，比 A 低 **9.4%**。三组均完成 1,806 帧，位置 ATE RMSE 为 2.605 / 2.640 / 2.617 cm。本结论限于本机、此序列和下面的配置；整段回放每组测量一次，冻结窗口另做五轮重复。

## 后端和共同设置

| 组别 | 构造 / 求值 | 求解 |
| --- | --- | --- |
| A：`native` | 原生逐观测因子、稠密 Jacobian 转稀疏正规方程 | Eigen SparseLU，原生阻尼与步长裁剪 |
| B：`ceres_scalar` | 每条观测一个 Ceres 因子 | Ceres LM，SuiteSparse `SPARSE_NORMAL_CHOLESKY` |
| C：`ceres_batch` | 本机 ClampSLAM 内 SPLBATCH 按完整参数支撑分组，共享计算 | 与 B 相同 |

A/B/C 使用共同的残差、权重、初始锚定和平方根边缘化先验。A 是统一模型后的原生求解流程；A/C 的比值不能解释为未经修改的 Noetic 程序的加速比。B/C 共享解析内核及求解选项，其同输入对照用于评估批处理收益。[实现说明](solver_comparison.md)

硬件为 AMD Ryzen 9 9950X（16 核 / 32 线程），系统 Ubuntu 24.04.2，GCC 13.3、Eigen 3.4、Ceres 2.1 / SuiteSparse 7.6.1。Release 构建使用 `cmake --build build/noetic --parallel 16`。基线为 `noetic` 标签 `0a79e9756c532cb2ba93990ff15edfc059367c58` 加本工作区修改；SPLBATCH 0.1.1 固定在 `e2892937c356b34d2873c91cea9b044a6c9d897a`。原始环境、二进制及配置 SHA-256 见文末清单。

数据来自用户下载的 `/home/jiadong/workspace/datasets/ntu/eee_03.zip`，已校验 ZIP 并解压到同级 `eee_03/`。bag 为 4,261,528,188 字节、181.353 秒。只使用水平雷达 `/os1_cloud_node1/points` 和 `/imu/imu`，外参核对包内 YAML；不融合相机、垂直雷达和真值。四阶样条、0.01 s 节点间隔、三帧窗口，`max_lidar_factor=8000`。配置见 [公共设置](../config/abc_common.yaml) 与 [NTU 传感器设置](../config/abc_ntuviral.yaml)。

## 完整回放：先看速度与精度

优化器使用 4 线程，bag 按 **1 倍速度**播放，三组均关闭快照写盘、回环、spline fitting、实时预算截断和外迭代提前终止。实测执行顺序为 B → C → A，各使用独立 ROS master，BLAS/MKL 单线程，CPU 亲和集合均为 0–31。

每帧共同执行三轮：

```text
优化一步 → 写回状态、IMU 传播 → 去畸变 → associate → 下一轮
```

每次求解内部迭代上限为 1。每轮处理最新两个点云，首帧处理整个三帧窗口；最后一轮也执行去畸变和关联。日志逐帧验证实际求解 / 去畸变 / 重新关联计数均为 **3 / 3 / 3**。三组各有 5,418 次求解调用；B/C 均在第 306 帧最后一轮因已收敛而执行 0 次内部迭代，仍完成对应的前端更新，其余调用均为 1 次。

| 后端 | 帧数 | 整帧均值 ms | 整帧 P95 ms | 最大积压包数 | 位置 ATE RMSE cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| A：native | 1,806 | 77.63 | 88.56 | 1 | 2.605 |
| B：ceres_scalar | 1,806 | 139.29 | 156.01 | 487 | 2.640 |
| C：ceres_batch | 1,806 | 70.31 | 80.22 | 1 | 2.617 |

三组 scan 时间戳逐帧一致，没有扫描缺口、空 LiDAR 约束帧、不可用解或异常退出。B 在约 10 Hz 输入下积压最多 487 包，最终将队列处理完；A/C 的最大积压为 1。B/C 的整帧平均时间比为 1.98，A/C 为 1.10。

整帧时间包含三次求解、公共最终代价诊断、边缘化、IMU 传播、去畸变、关联及其他前端工作，不含等待输入、帧 CSV 自身写盘和结束后的自动退出等待。公共最终代价诊断每次约 9.4 ms，每帧合计约 28 ms；这是本比较模式的测量开销，整帧结果包含它。冻结窗口的后端核心时间则排除此项。全回放时各后端状态和后续关联会变化，因此其求解耗时不能当作严格同输入比较。

### 真值精度的计算

从 `/leica/pose/relative` 读取全部 2,990 个真值 **header 时间戳**，三组在同一批时间戳上求实际 SO(3) + R³ 样条位姿；保存的控制点没有直接当作轨迹点。采用包内 `T_Body_Prism` 平移 `[-0.293656, -0.012288, -0.273095]` m，将估计变换到棱镜位置，再分别做无尺度 SE(3) 位置对齐。不拟合时间偏移。棱镜补偿与刚体对齐参照 [NTU 官方评估教程](https://ntu-aris.github.io/ntu_viral_dataset/evaluation_tutorial.html)。

共同真值覆盖率为 100%，时间段为 `1609060363.469352`–`1609060514.223515`，共 150.754 秒。真值从 bag 开始约 28.7 秒后才出现，因此 ATE 不覆盖最前面的静止初始化；完整处理情况由上述全序列日志单独验证。

| 后端 | RMSE cm | P95 cm | 最大误差 cm |
| --- | ---: | ---: | ---: |
| A | 2.605 | 4.770 | 7.151 |
| B | 2.640 | 4.795 | 7.075 |
| C | 2.617 | 4.772 | 7.144 |

三组在本序列上的精度接近。尚未做跨序列统计或多次完整回放，不据此宣称普遍精度等价。[轨迹与误差图 PNG](../build/abc_eee03_accuracy/trajectories.png) / [PDF](../build/abc_eee03_accuracy/trajectories.pdf)。

## 相同输入窗口：一次优化的耗时

从独立 A 全序列回放中每 18 帧采集一次首次外迭代输入，得到 101 个窗口（第 1–1,801 帧，跨度约 180 秒）。A/B/C 逐窗口使用相同快照，包括初值、关联、权重和先验，且校验输入摘要一致。每个线程配置测 5 轮，各后端每配置 505 个正式样本，总计 6,060 个样本。

窗口包含 31–35 个控制点、116–135 个 IMU 观测、4951–8001 个 LiDAR 观测。各后端使用独立进程，每轮全量预热一次、轮换顺序、保持相同 CPU 亲和集合；BLAS/MKL 单线程。

表中单位为 **ms，均值 / P95**。包含状态重置、问题构造、求解、析构；不含快照读取、独立最终代价诊断、边缘化和前端去畸变/关联。固定关联测试只衡量一次优化调用，不替代完整回放。

| 优化线程 | A 均值 / P95 | B 均值 / P95 | C 均值 / P95 | B/C 配对时间比中位数 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.63 / 17.86 | 32.90 / 35.63 | 8.97 / 9.53 | 3.70 |
| 4 | 7.75 / 8.55 | 29.80 / 32.81 | 5.48 / 6.21 | 5.42 |
| 8 | 6.80 / 8.18 | 28.49 / 31.53 | 4.66 / 5.33 | 6.16 |
| 16 | 6.41 / 7.55 | 27.78 / 31.31 | 4.56 / 5.09 | 6.14 |

4 线程时，C 的核心平均耗时比 B 低 81.6%，比 A 低 29.3%。B/C 的平均残差块数从 7692.4 降到 62.2，标量残差行数保持一致。

| 4 线程阶段均值 ms | B | C |
| --- | ---: | ---: |
| 问题构造 | 2.772 | 1.266 |
| Ceres 求值 | 4.256 | 2.398 |
| Ceres 线性阶段 | 16.319 | 1.080 |
| Solve 总计（包含上两项） | 24.810 | 4.135 |
| 问题析构 | 2.206 | 0.078 |

最大的节省出现在 Ceres Summary 记录的线性阶段；该阶段包含线性求解器内部处理，不能把整个差值都归为稀疏分解。构造和求值也有所减少。分项与 Solve 总计不可重复相加。

4 线程进程峰值 RSS 为 A 116.5 MiB、B 99.1 MiB、C 91.6 MiB，包含载入的全部快照、运行时和优化内存。

| 相同输入数值检查 | 最大差异 |
| --- | ---: |
| A 与 B/C 的 H（相对） | 1.41e-15 |
| A 与 B/C 的梯度（相对） | 2.34e-12 |
| A 与 B/C 的初始代价（相对） | 4.16e-13 |
| B/C 求解状态，分别检查 1 和 8 次迭代 | 2.41e-11 |
| 正式计时 B/C 的最终代价（相对） | 7.4e-13 |

全部 101 个窗口通过检查；8 次迭代只用于独立数值验证，正式计时和主回放均采用单步上限。状态差取控制点平移、SO(3) log 旋转及 bias 的最大向量范数。

4 线程平均最终代价为 A 335.25243、B 158.23268、C 158.23268；显著代价上升的样本数分别为 5 / 0 / 0（各 505 个，阈值为 `1e-9 * max(1, |initial_cost|)`），全部解可用。A 的阻尼和步长裁剪与 Ceres 不同，相同单步预算不表示三组收敛程度相同。

A 的这 5 个样本均为首个启动窗口的重复：代价从 14,493.78 上升到 32,372.77；B/C 拒绝该步，代价保持 14,493.78。该窗口保留在全部统计中，对平均代价影响较大；三组最终代价中位数均约 14.4390。轨迹精度仍应看共同真值的 ATE，不能由冻结问题的平均代价推断。

## 保持单步调度的 Ceres 修复

最初 B 每次重建 Problem 并调用一次 `Solve`，都会把信赖域半径重置为默认 `1e4`，丢失前一次 LM 调整结果。在 `eee_03` 上，B 从第 152 帧开始没有 LiDAR 约束；该次运行中断并保留在 `build/abc_eee03_full/failure_review.json`，后续失败状态的低耗时没有用于性能结论。

现在 B/C 将 Ceres 返回的最终信赖域半径传到下一次调用，跨外迭代和帧连续传递，**每次内部迭代上限仍然是 1**。CSV 同时记录输入和输出半径，回放脚本检查相邻调用连续性。这里接续的是信赖域半径，没有声称保留了 Ceres 所有内部优化状态。

本报告的完整回放均使用修复后的同一二进制。B/C 均从 `1e4` 开始，在第 32 次调用的输入达到 Ceres 默认上限 `1e16`，其后保持；5,418 次调用中有 5,387 次以该半径开始（99.43%）。离线冻结窗口统一使用 `1e16`，用于比较该稳定阶段的单步计算；每个窗口独立重置，未混用实时半径传递和离线独立问题。

## 验证和复现

C++ CTest 已通过：包括残差/Jacobian/正规方程一致性、有限差分、先验与独立 QR 消元比较、快照完整性、线程与连续单步求解检查。新增 Python 精度评估的三个测试通过，覆盖样条相位与旋转、刚体对齐保持尺度、OpenCV 格式棱镜标定。

另外，将 Python 样条求值与仓库 `basalt::Se3Spline<4>::pose` 独立 C++ 程序在全部 2,990 个真值时刻比较：最大位置差 `7.32e-15` m，最大旋转差 `7.77e-16` rad。该检查确认评估没有把控制点当采样点或引入节点相位错误，结果位于 `build/ntu_eval_reference/validation.json`。

正式完整回放的命令如下；复跑需使用新的输出目录：

```bash
source devel/setup.bash  # zsh 使用 devel/setup.zsh
/usr/bin/python3 scripts/run_abc_replay.py \
  --dataset ntuviral \
  --bag /home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag \
  --output build/abc_eee03_single_step \
  --backends ceres_scalar ceres_batch native \
  --duration 0 --rate 1 --threads 4 --outer-iterations 3 \
  --timeout 1800 --no-snapshots

/usr/bin/python3 scripts/evaluate_ntuviral.py \
  --bag /home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag \
  --prism-calibration /home/jiadong/workspace/datasets/ntu/eee_03/leica_prism.yaml \
  --replay build/abc_eee03_single_step --output build/abc_eee03_accuracy
```

本次评估实际使用 `--ground-truth build/abc_eee03_inputs/ground_truth.csv`，该 CSV 已从同一 bag 的真值消息提取；与上面直接读 bag 的入口计算相同。

同输入采集与计时使用以下命令。采集只有 A 写盘，该次运行的整帧耗时不加入正式回放表。冻结计时开始前已停止 ROS 回放，计时期间未同时编译或执行其他测量。

```bash
/usr/bin/python3 scripts/run_abc_replay.py \
  --dataset ntuviral \
  --bag /home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag \
  --output build/abc_eee03_capture --backends native \
  --duration 0 --rate 1 --threads 4 --outer-iterations 3 \
  --timeout 600 --snapshot-stride 18 --snapshot-limit 200

/usr/bin/python3 scripts/run_abc_benchmark.py \
  --snapshots build/abc_eee03_capture/snapshots --output build/abc_eee03_frozen \
  --threads 1 4 8 16 --iterations 1 --ceres-initial-radius 1e16 \
  --rounds 5 --max-snapshots 101 --validate 101
```

可追溯记录：

- [汇总 JSON](solver_comparison_eee03.json)：本报告数值、环境、检查和原始记录路径。
- `build/abc_eee03_single_step/`：`replay_manifest.json`、`environment.json`、三组 `optimization.csv` / `frames.csv` / `trajectory/`。
- `build/abc_eee03_accuracy/`：`accuracy.json`、每组误差 CSV、采样位姿 TUM、轨迹与误差 PNG/PDF。
- `build/abc_eee03_frozen/`：每次独立进程的 CSV/log、`all.csv`、`summary.json`、`validation.log`、含实际命令与代码/二进制摘要的 `manifest.json`。
- `build/abc_eee03_capture/`：独立 A 回放和 101 个相同输入问题快照。

构建和原始实验输出位于被 Git 忽略的 `build/`，本 Markdown 和汇总 JSON 保存在 `docs/`。旧 R3LIVE 的 A=1、B/C=8 结果仍保留为历史记录，没有与本次统一单步结果合并。
