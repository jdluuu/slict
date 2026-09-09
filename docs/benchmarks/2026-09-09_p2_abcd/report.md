# P2 四后端完整冻结窗口实验

2026-09-09，重新测量当前工作区 P2 版本的四个后端，仅使用 1/4/8 个优化线程。本报告不混入旧 P1 或其他历史版本的耗时。

## 配置与计时

硬件：AMD Ryzen 9 9950X 16-Core Processor；CPU 亲和集合为 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]。使用 Release 构建，同一份可执行文件和共享库；构建并行度为 16，三个 CTest 均通过。具体版本、源文件和二进制 SHA-256 保存在 manifest.json 和 audit.json。

- 输入：NTU VIRAL eee_03 全序列中此前采集的 101 个冻结窗口。四组每个窗口使用相同初值、关联、权重和先验。
- 计算预算：每次优化最多一步，1/4/8 线程，每个配置五轮、505 个正式样本，总计 6,060 个样本。
- 每个后端/线程/轮次使用独立进程，先完整预热一次，后端执行次序轮换。BLAS/MKL 与 Eigen 内部线程为 1，具体 OpenMP 环境见清单。正式测量在构建、CTest 和独立数值验证结束后执行。
- Ceres 在每个冻结窗口上独立重置初始信赖域半径为 1e4，与旧 P2 对照实验一致；不使用在线半径传递。早期 A/B/C 稳定阶段实验使用过 1e16，其历史数据不能混作本轮对照。
- C/D 使用 P1＋P2 的串行建批；A/B 也使用同一当前版本。P2 不改变 A/B 的因子处理方式，未加入直接 H/g 累加。
- total_ms 包括状态复制、构造、求解和末尾析构，不包含公共最终代价诊断、快照 I/O、边缘化或前端。本次没有重新做在线整包回放。

## 正式结果

单位为 ms，表中为均值 / P95。

| 优化线程 | native | native_batch | ceres_scalar | ceres_batch |
|---:|---:|---:|---:|---:|
| 1 | 17.075 / 18.746 | 7.781 / 8.435 | 32.998 / 36.241 | 8.269 / 8.862 |
| 4 | 7.613 / 8.446 | 6.386 / 6.993 | 29.449 / 32.360 | 4.468 / 4.922 |
| 8 | 6.616 / 8.200 | 6.342 / 7.459 | 28.326 / 31.204 | 3.955 / 4.471 |

完整阶段均值、内存和均值时间比见 [汇总 CSV](../../solver_speed_summary.csv)。分项中 solve 包含内部求值、组装和求解，不能重复相加；Ceres 未独立计时的组装和更新项留空，其线性阶段不等同于纯矩阵分解。

## 数值检查与边界

独立进程检查全部 101 个窗口，比较四组 H/g/初始代价，并检查 A/D、B/C 在 1 和 8 次迭代上限时的状态；8 步仅用于数值验证，不计入正式性能统计。最大 H 相对误差 1.41e-15，g 相对误差 2.34e-12，初始代价相对误差 4.16e-13。B/C 最大状态差 1.24e-12，A/D 最大状态差 2.87e-11。

正式样本按窗口、轮次和线程数检查输入配对完整性，所有解 usable。A/D 与 B/C 的最终代价相对误差均通过 1e-7 阈值；各线程的最大误差和配对时间比中位数见 summary.json。

| 线程 | native 代价上升样本 | native_batch 代价上升样本 | ceres_scalar 代价上升样本 | ceres_batch 代价上升样本 |
|---:|---:|---:|---:|---:|
| 1 | 5 | 5 | 0 | 0 |
| 4 | 5 | 5 | 0 | 0 |
| 8 | 5 | 5 | 0 | 0 |

代价上升阈值为 1e-9 × max(1, |initial_cost|)，所有样本均保留。A/D 的原生阻尼和步长裁剪与 B/C 的 LM 接受策略不同，相同单步预算不代表达到相同收敛程度。此处报告冻结窗口效率，不推断整帧加速、轨迹精度或统计显著性。

## 保存与复现

- [全部 6,060 条原始样本](all.csv)
- [统计与配对比较](summary.json)
- [运行命令、环境和二进制清单](manifest.json)
- [输入哈希、逐轮均值与数值验证摘要](audit.json)
- [独立数值验证日志](validation.log)、[CTest 日志](ctest.log)、[动态链接记录](linkage.txt)

以上文件保存在 docs/benchmarks/，不依赖 build/ 中的结果副本。输入快照与可执行文件仍位于本地 build/、devel/；归档保存其路径和哈希，不包含输入快照或二进制本体。尚未自动提交 Git。

使用新的输出目录复跑：

```bash
cmake --build build/noetic --parallel 16 --target slict_solver_benchmark slict_solver_comparison_test slict_splbatch_ingestion_test slict_splbatch_incremental_test
ctest --test-dir build/noetic --output-on-failure -R '^slict_(solver_comparison|splbatch_ingestion|splbatch_incremental)$'
python3 scripts/run_abc_benchmark.py --snapshots build/abc_eee03_capture/snapshots --output build/p2_abcd_repeat --backends native native_batch ceres_scalar ceres_batch --threads 1 4 8 --iterations 1 --ceres-initial-radius 1e4 --rounds 5 --max-snapshots 101 --validate 101
```
