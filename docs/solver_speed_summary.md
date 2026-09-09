# 后端速度汇总 CSV

[solver_speed_summary.csv](solver_speed_summary.csv) 是 **2026-09-09 重新实测**的 P2 四后端结果，只含本轮 1/4/8 线程数据。共 12 行配置，来源为 6,060 个正式样本；每组 101 个 eee_03 冻结窗口、5 轮、505 个样本，内部迭代上限为 1。

[完整实验报告](benchmarks/2026-09-09_p2_abcd/report.md) 包含配置、均值/P95、数值检查和复现命令。

另已完成四后端、1/4/8 线程的 **eee_03 完整在线回放**：见 [整帧耗时与最终精度报告](benchmarks/2026-09-09_p2_abcd_full/report.md)、[完整耗时 CSV](benchmarks/2026-09-09_p2_abcd_full/timing_summary.csv)、[最终精度 CSV](benchmarks/2026-09-09_p2_abcd_full/accuracy_summary.csv)。本页和原汇总 CSV 仍保留冻结窗口结果。

## 选择对照

- 所有行的 `experiment=p2_abcd_20260909_frozen`、`version=p2`，表示使用同一当前 P2 构建，不混入 P1/before 或历史 A/B/C 测量。
- A=`native`：逐观测原生因子＋原生组装和 SparseLU；D=`native_batch`：SPLBATCH 批量因子＋原生组装和 SparseLU。
- B=`ceres_scalar`：每观测一个 Ceres 残差块＋Ceres LM；C=`ceres_batch`：SPLBATCH 批量因子＋Ceres LM。
- C/D 使用 P1 支撑查找优化＋P2 轻量输入，建批仍串行。A/B 不使用这条批处理构造路径，但也在同轮、同程序中重新计时。
- Ceres 在每个冻结窗口上独立重置初始信赖域半径为 `1e4`，与旧 P2 实验一致。早期 A/B/C 的稳定阶段实验使用过 `1e16`，不能混作本轮对照。

不同实验不能混合计算后端加速比。原生与 Ceres 使用不同阻尼、步长和接受策略，单步时间比不表示相同收敛程度的加速比。此 CSV 不包含整帧回放、带探针诊断或已回退的并行建批实验；这些历史结果仍保存在各自 JSON/Markdown 中。

## 列含义与计时口径

- `*_mean_ms`：平均时间，单位毫秒。`total_p95_ms` 为总耗时 P95，直接保留来源 JSON 的统计。
- `total_mean_ms` 包括状态复制、问题构造、求解和末尾清理，不包括公共最终代价诊断、快照 I/O、边缘化及前端。
- `solve_mean_ms` 包含求值、组装、线性求解等内部工作，不能再与这些子项相加。
- 原生 `other_mean_ms = total - reset - build - evaluate - assemble - linear - update - destroy`，是差值项，不是独立探针测量。
- Ceres 的 `assemble_mean_ms`、`update_mean_ms`、`other_mean_ms` 留空，表示未独立计时。Ceres `linear_mean_ms` 是 Summary 的线性阶段，不等同于纯矩阵分解时间；`evaluate_mean_ms` 包含残差与 Jacobian 求值时间，与原生一次 r/J 求值的工作量不必相同。
- `speedup_vs_native_mean_ratio`：同实验、同线程 native 平均总时间 / 本行平均总时间；大于 1 表示快于 native。
- `speedup_vs_1thread_mean_ratio`：同实验、同后端、同版本的单线程平均总时间 / 本行平均总时间。
- `time_reduction_vs_native_pct`：相对同实验、同线程 native 的耗时下降百分数，正值表示耗时减少。例如 16.2 表示 16.2%。以上速度比都是均值之比，不是逐窗口配对比值的中位数。
- `mean_actual_iterations` 保留实际迭代均值；`iteration_limit=1` 不保证每次调用都更新一步。
- `ceres_initial_trust_region_radius` 是 Ceres 的初始信赖域半径，对 A/D 留空。
- `peak_rss_mib` 为来源统计的进程峰值内存，包含加载的快照和运行时。
- `source_json` 与 `source_raw_csv` 均为仓库相对路径。均值和样本数已与原始 CSV 核对，缺失分项没有填成零。

## 本轮数据保存位置

- [全部 6,060 条原始样本 CSV](benchmarks/2026-09-09_p2_abcd/all.csv)
- [统计和配对比较 JSON](benchmarks/2026-09-09_p2_abcd/summary.json)
- [运行命令、环境及代码/二进制哈希](benchmarks/2026-09-09_p2_abcd/manifest.json)
- [输入哈希、逐轮统计与验证摘要](benchmarks/2026-09-09_p2_abcd/audit.json)
- [数值验证日志](benchmarks/2026-09-09_p2_abcd/validation.log)、[CTest 日志](benchmarks/2026-09-09_p2_abcd/ctest.log)

结果归档位于 `docs/benchmarks/`，不再只依赖被 Git 忽略的 `build/` 副本。本次没有在线整包回放；所有正式样本均保留，代价上升次数和配对误差见实验报告。尚未自动提交 Git；输入快照和二进制本体仍在本地 build/devel 路径，归档记录其路径和哈希。

## 历史数据

- [P2 持久化结果](splbatch_light_binding_results.json) / [报告](splbatch_light_binding_results.md) / [逐样本 CSV](../build/p2_eee03_frozen/all.csv)
- [A/B/C 持久化结果](solver_comparison_eee03.json) / [报告](solver_comparison_eee03.md) / [逐样本 CSV](../build/abc_eee03_frozen/all.csv)

以上历史数据不参与当前汇总的加速比计算。汇总 CSV 是本轮测量的静态导出，后续有新实验时应按实验标识追加或重新生成。
