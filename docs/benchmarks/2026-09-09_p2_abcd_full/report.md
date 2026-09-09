# P2 四后端：eee_03 完整回放耗时与精度

2026-09-09，当前 P2 版本在 NTU VIRAL eee_03 全序列上运行四个后端，优化线程分别为 1/4/8。共 12 次独立回放，每个配置完整运行一次；没有把冻结窗口的五轮重复算作完整回放重复。原冻结窗口报告和数据保持不变。

## 共同设置

bag 时长 181.353 s，使用水平 LiDAR /os1_cloud_node1/points 和 /imu/imu。每帧三轮，每轮执行一步优化、状态更新、去畸变和重新关联。窗口长度三帧，四阶样条，节点间隔 0.01 s，最多 8000 个 LiDAR 因子。

P2 的轻量输入与支撑查找用于 native_batch / ceres_batch，建批串行。native / ceres_scalar 同样使用本轮当前构建。四者使用共同模型和边缘化先验；A/D 采用原生阻尼、裁剪和 SparseLU，B/C 使用 Ceres LM / SuiteSparse Cholesky。

bag 均以 1 倍速播放，关闭快照写盘、回环、样条拟合、实时预算截断和外迭代提前结束。各组独立 ROS master，按顺序运行，未并行执行测量或编译。CPU 为 Ryzen 9 9950X，亲和集合 0–31，BLAS/MKL 单线程。构建使用 --parallel 16。

每次完整回放的 Ceres 半径从 1e4 开始，随后跨调用传递；这与冻结窗口每个输入独立重置半径的策略不同。逐调用 CSV 和回放脚本核对半径连续性。

## 整帧耗时与实时性

| 线程 | 后端 | 帧数 | 整帧均值 ms | P95 ms | 累计帧处理 s | 最大积压包数 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | native | 1806 | 111.098 | 126.129 | 200.643 | 219 |
| 1 | native_batch | 1806 | 81.497 | 92.073 | 147.184 | 1 |
| 1 | ceres_scalar | 1806 | 157.729 | 175.900 | 284.859 | 625 |
| 1 | ceres_batch | 1806 | 82.026 | 93.003 | 148.139 | 1 |
| 4 | native | 1806 | 77.233 | 87.888 | 139.483 | 1 |
| 4 | native_batch | 1806 | 73.862 | 84.992 | 133.395 | 1 |
| 4 | ceres_scalar | 1806 | 138.932 | 155.573 | 250.912 | 479 |
| 4 | ceres_batch | 1806 | 67.279 | 77.137 | 121.506 | 1 |
| 8 | native | 1806 | 73.074 | 83.592 | 131.971 | 1 |
| 8 | native_batch | 1806 | 72.675 | 82.515 | 131.251 | 1 |
| 8 | ceres_scalar | 1806 | 136.263 | 152.522 | 246.090 | 458 |
| 8 | ceres_batch | 1806 | 65.081 | 74.502 | 117.536 | 1 |

整帧时间由 frames.csv 记录，包括三轮优化、公共最终代价诊断、边缘化及前端等工作，不包含等待输入、frames.csv 本身写盘和结束后的自动退出等待。累计帧处理时间是各帧 frame_ms 之和，并非 ROS 进程墙钟时间或整台机器的 CPU 时间。

## 后端与公共开销

| 线程 | 后端 | 核心均值 ms/调用 | 核心 P95 ms/调用 | 后端含公共项 ms/调用 | 代价诊断 ms/调用 | 边缘化 ms/帧 | 剩余整帧工作 ms/帧 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | native | 18.517 | 21.682 | 30.196 | 9.377 | 5.774 | 20.511 |
| 1 | native_batch | 8.523 | 9.504 | 20.325 | 9.475 | 5.821 | 20.523 |
| 1 | ceres_scalar | 33.961 | 38.063 | 45.610 | 9.309 | 5.873 | 20.900 |
| 1 | ceres_batch | 8.839 | 9.807 | 20.548 | 9.393 | 5.790 | 20.382 |
| 4 | native | 8.288 | 9.612 | 19.062 | 9.438 | 2.835 | 20.048 |
| 4 | native_batch | 7.043 | 8.095 | 17.890 | 9.511 | 2.859 | 20.191 |
| 4 | ceres_scalar | 28.531 | 32.166 | 39.323 | 9.396 | 3.021 | 20.964 |
| 4 | ceres_batch | 4.776 | 5.421 | 15.600 | 9.454 | 2.934 | 20.479 |
| 8 | native | 6.752 | 8.421 | 17.487 | 9.549 | 2.359 | 20.613 |
| 8 | native_batch | 6.561 | 7.706 | 17.274 | 9.516 | 2.388 | 20.854 |
| 8 | ceres_scalar | 27.685 | 31.367 | 38.367 | 9.443 | 2.539 | 21.163 |
| 8 | ceres_batch | 4.099 | 4.702 | 14.762 | 9.486 | 2.343 | 20.795 |

核心时间包括 reset/build/solve/destroy。后端含公共项时间在核心上增加最终代价诊断、输入转换/状态回写和边缘化；边缘化仅在每帧最后一轮执行。剩余整帧工作按 frame_ms 减去该帧全部后端含公共项时间计算，包含前端、输入摘要、日志和未单独计时的调用开销，不应全部解释为关联时间。

构造、求值、H/g 组装、线性阶段、更新和清理的完整均值/P50/P95/最大值/累计值保存在 summary.json 的 detailed_stage_statistics；solve 包含内部子阶段，不可重复相加。Ceres CSV 中 assemble/update 的零是未单独计时，不能理解为零成本。

timing_summary.csv 还包含进程墙钟时间、累计核心时间、累计含公共项后端时间、超 100 ms 帧数和内存峰值。processing_capacity_fps = 1000/frame_mean_ms，只表示按平均处理时间推算的处理能力，不是 bag 实际输出频率。process_wall_s 包含 ROS 启动、播放、等待与退出开销。

## 最终轨迹精度

使用该 bag 内 /leica/pose/relative 的全部 2,990 个 header 时间戳，在保存的 SO(3)+R3 样条上求位姿；加入数据集 T_Body_Prism 的棱镜杆臂补偿，然后独立作无尺度 SE(3) 位置对齐。不拟合时间偏移，也不把控制点直接当作轨迹采样点。

12 组使用完全相同的真值时间戳和位置，覆盖率均为 100%。共同真值区间为 1609060363.469352007–1609060514.223515034，长度 150.754 s。真值晚于 bag 起点约 28.7 s，因此 100% 表示覆盖全部已有真值，不表示真值覆盖 bag 开头。

| 线程 | 后端 | ATE RMSE cm | 中位数 cm | P95 cm | 最大误差 cm |
|---:|---|---:|---:|---:|---:|
| 1 | native | 2.600 | 2.098 | 4.717 | 7.006 |
| 1 | native_batch | 2.613 | 2.110 | 4.737 | 6.936 |
| 1 | ceres_scalar | 2.623 | 2.111 | 4.770 | 7.029 |
| 1 | ceres_batch | 2.637 | 2.155 | 4.767 | 6.981 |
| 4 | native | 2.611 | 2.097 | 4.759 | 7.194 |
| 4 | native_batch | 2.618 | 2.111 | 4.730 | 7.179 |
| 4 | ceres_scalar | 2.632 | 2.108 | 4.753 | 7.191 |
| 4 | ceres_batch | 2.661 | 2.163 | 4.836 | 7.039 |
| 8 | native | 2.605 | 2.110 | 4.719 | 6.960 |
| 8 | native_batch | 2.615 | 2.099 | 4.721 | 7.041 |
| 8 | ceres_scalar | 2.611 | 2.102 | 4.758 | 7.136 |
| 8 | ceres_batch | 2.639 | 2.163 | 4.758 | 7.195 |

这里的最终精度指完整回放保存轨迹在全部共同真值时刻的 ATE，不是仅最后一个位姿的终点误差，也未评估旋转真值误差。每个配置仅完整回放一次，不依据微小 ATE 差异或单次耗时差异宣称统计显著性。

- [1 线程轨迹及误差图](t1_accuracy/trajectories.png)
- [4 线程轨迹及误差图](t4_accuracy/trajectories.png)
- [8 线程轨迹及误差图](t8_accuracy/trajectories.png)

- [各线程整帧处理时间与队列曲线](frame_timing.png)
- [整帧均值与 ATE 汇总图](speed_accuracy.png)
- [优化核心耗时对比 PNG](optimization_timing.png) / [PDF](optimization_timing.pdf)：每次调用的核心耗时均值，每组 5,418 次调用。颜色区分后端；斜线表示构造 build，黑色网格表示求解 solve，反斜线表示其余开销（reset、destroy 及阶段计时空隙，按 total − build − solve 计算）。柱顶为总毫秒数，各段高度表示阶段耗时，柱内不标百分比。solve 已含求值、组装、线性求解和更新，不重复堆叠其子阶段。使用 `python3 scripts/plot_optimization_timing.py docs/benchmarks/2026-09-09_p2_abcd_full` 重新生成。

处理时间曲线采用 20 帧滑动平均，队列曲线保留原值；表格和 CSV 统计均使用未经平滑的原始计时。图中 100 ms 虚线对应约 10 Hz 输入的单帧时间预算。

## 完整性与可复查结果

每组均处理 1,806 帧、5,418 次优化，共 21,672 帧、65,016 次优化。所有 12 组扫描时间戳一致、扫描连续，每帧求解/去畸变/关联计数均为 3/3/3。没有不可用解、空 LiDAR 约束帧、超时或节点异常退出，最终排空输入队列。

各配置的零迭代调用数、代价上升调用数保存在 timing_summary.csv。每次最多一步允许 Ceres 因已收敛而不执行更新；原生与 Ceres 的接受策略不同。完整回放后状态和关联会发生分化，所以这里的后端核心耗时不再是严格同输入比较。相同输入比较仍看原冻结窗口报告。

- [整帧和后端耗时汇总 CSV](timing_summary.csv)
- [最终位置精度汇总 CSV](accuracy_summary.csv)
- [完整分项统计 JSON](summary.json)
- [输入与程序哈希、执行命令](suite_manifest.json)
- [归档文件哈希](archive_sha256.json)
- [现有冻结窗口报告](../2026-09-09_p2_abcd/report.md)

t1/t4/t8 下分别保留四个后端的 frames.csv、optimization.csv 和 trajectory/spline_log.csv，以及 replay_manifest.json。对应的 t*_accuracy 目录保存 accuracy.json、逐点误差 CSV、机体轨迹 TUM 和轨迹/误差图 PNG/PDF。ground_truth.csv 是本次从原 bag 提取的 Leica 真值。

原始 ROS 日志保留在 build/p2_abcd_20260909_full，各组 CSV、最终样条、精度与清单另外归档到 docs/benchmarks/2026-09-09_p2_abcd_full，不会随 build 结果清理而丢失。输入 bag 和可执行文件本体未复制，保存了位置和哈希。尚未自动提交 Git。

## 复现

```bash
source devel/setup.bash
python3 scripts/run_abc_replay.py --dataset ntuviral \
  --bag /home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag \
  --output build/p2_full_repeat_t4 \
  --backends native native_batch ceres_scalar ceres_batch \
  --threads 4 --outer-iterations 3 --duration 0 --rate 1 --timeout 1800 --no-snapshots
/usr/bin/python3 scripts/evaluate_ntuviral.py \
  --bag /home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag \
  --prism-calibration /home/jiadong/workspace/datasets/ntu/eee_03/leica_prism.yaml \
  --replay build/p2_full_repeat_t4 --output build/p2_full_repeat_t4_accuracy \
  --backends native native_batch ceres_scalar ceres_batch
```

1/8 线程同样执行，替换线程数并使用新的输出目录；各次测量顺序与实际命令以 suite_manifest.json 为准。
