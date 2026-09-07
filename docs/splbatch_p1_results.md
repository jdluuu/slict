# SPLBATCH P1：支撑查找缓存

2026-09-07。仅实施 P1：保持逐条 `AddObservation(binding, observation)`，由 SPLBATCH 自动识别支撑。P2 的 Binding 容量预设、轻量存储以及 P3/P4 均未加入。

本页记录 P1 完成时的版本及测量。后续轻量 Bind 输入与直接数组查找已另行实现，见 [轻量 Bind 结果](splbatch_light_binding_results.md)。

## 实现范围

- `problem_batcher.hpp`：`last_batch_` 保存索引；连续支撑完整比较命中后直接追加。其他输入按哈希查候选，再比较全部有序参数地址、尺寸和 `support_signature`。首次支撑沿用完整参数校验，批次按首次出现顺序保存。
- `batch_binding.hpp`：增加 `detail::SameBatchSupport` 和 `detail::HashBatchSupport`，直接读取输入数组，不创建拥有动态数组的临时 `BatchKey`。比较先检查三个数组的长度，哈希碰撞不能决定相等。
- 支撑描述由 channel 私有的 `PendingBatch::binding` 拥有，不借用调用方临时 Binding 的数组。每次仍使用本次 Binding/context 执行 `PrepareObservation`。
- 预计算失败不修改存储；loss scale 追加后若 prepared observation 追加失败，则撤回 scale。新批次索引分配失败会撤回 pending batch。只有成功添加后才更新 `last_batch_`。
- Commit 清空当前 channel 的表和索引。缓存不跨 channel、建批周期或优化调用保存。

首次建批仍有存储与索引分配，观测数组扩容仍有分配；每条观测仍需要 Binding 构造、少量支撑比较和预计算。P1 消除的是重复 Key 构造与完整参数校验，不能把整个添加过程称为 O(支撑数)。

`src/comparison/evaluators.h`、SLICT 的 Bind 助手、native/Ceres 后端、batch 求值与 Jacobian 写入实现均未修改。

## IncrementalProblemBatcher 的兼容边界

普通 `EvaluatorChannel` 与 `IncrementalEvaluatorChannel` 是独立路径。此次 `incremental_problem_batcher.hpp` 内容未变；`batch_binding.hpp` 去掉新增辅助函数后，与修改前逐字节相同。因此原 `Key()`、`BatchKey`、`BatchKeyHash`、参数校验和拥有存储的 Binding 接口不变。

增量路径继续按 `BatchKey + partition + binding_revision` 判断批次。数值更新刷新 prepared slot；支撑或语义变化仍迁移观测并更新后端因子。P1 没有在 Binding 内保存可能过期的 Key，也没有把普通 channel 的索引用于增量缓存。

## 验证方法

新增 `slict_splbatch_ingestion` CTest 覆盖：

- 160 个支撑、480 条观测的有序和随机输入，参考分组使用保留的旧 `Key()`；检查首次批次顺序、批内顺序、不同 context 和逐条 loss scale。
- 临时输入数组销毁、相同地址开始新建批周期、不同 channel，以及参数数值更新后重新求值。
- 参数地址、顺序、尺寸、签名及数组长度变化；空指针、重复指针、非法尺寸和非法 loss scale。
- 人工构造两个完整哈希相同的有效支撑，交错添加后验证没有误合批。
- 在首次添加、连续命中、表命中、新支撑添加过程中，逐个注入 C++ `new` 分配失败，验证批次数、观测数以及恢复后的残差和 loss scale。另行注入 PrepareObservation 异常。
- 已有观测数组容量足够时，连续命中和表命中的添加路径均不调用 C++ `new`。这不代表整个 SLICT Bind 或全部构造零分配。
- 普通 compressed channel 的代价、H/g；增量同支撑刷新、跨支撑迁移、partition/revision 分别改变、删除后重新添加。

上游 `tests/test_incremental.cpp` 从同一 vendored 来源原样引入，作为 `slict_splbatch_incremental` CTest。它覆盖数值刷新、共享存储、支撑合并、失效依赖、窗口过期、sealed append、压缩注册，以及 1,500 步与完整扫描参考实现比较的增量生命周期。

原 `slict_solver_comparison` 继续检查 A/B/C/D 的残差、Jacobian、有限差分、先验、固定参数、仅 IMU/仅 LiDAR/空问题以及 1/8 步状态。

以上三个 CTest 均通过。新增 ingestion/增量迁移用例也通过 AddressSanitizer 和 UndefinedBehaviorSanitizer。沙箱下 LeakSanitizer 不支持当前 ptrace 环境，因此该次运行关闭泄漏检查；这项结果不包含泄漏检测。日志位于 `build/p1_ctest.log` 和 `build/p1_sanitizers_test.log`。

## 无探针正式结果

同机、同窗口、同轮次重新运行的均值如下，单位 ms。各表每行的每个版本均有 505 个样本。

| 线程 | D 构造：优化前 → P1 | 构造下降 | D 核心总时间：优化前 → P1 | P1 总时间 P95 | A 核心总时间 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.171 → 0.873 | 25.5% | 8.676 → 8.304 | 9.163 | 17.488 |
| 4 | 1.197 → 0.878 | 26.6% | 7.509 → 6.977 | 7.821 | 8.411 |
| 8 | 1.237 → 0.912 | 26.2% | 7.362 → 7.032 | 8.202 | 7.482 |
| 16 | 1.244 → 0.952 | 23.5% | 7.426 → 7.050 | 8.042 | 6.635 |

| 线程 | C 构造：优化前 → P1 | 构造下降 | C 核心总时间：优化前 → P1 |
|---:|---:|---:|---:|
| 1 | 1.101 → 0.781 | 29.1% | 9.133 → 8.690 |
| 4 | 1.294 → 1.009 | 22.0% | 5.509 → 5.233 |
| 8 | 1.361 → 1.174 | 13.7% | 4.849 → 4.832 |
| 16 | 1.295 → 1.025 | 20.8% | 4.750 → 4.491 |

D 的构造在四种线程配置下降 23.5%–26.6%。4 线程下降 0.318 ms；核心总时间下降 7.1%。C 同样受益，但 8 线程核心总时间仅下降 0.35%，不据此声称稳定的整体加速。16 线程 P1 后 D 仍比本轮 A 慢约 6.2%，P1 没有解决所有高线程开销。

4 线程 D 分项：

| 指标 | 优化前均值 / P95 | P1 均值 / P95 |
|---|---:|---:|
| 构造 | 1.197 / 1.363 | 0.878 / 1.009 |
| 求值 | 1.737 / 2.218 | 1.705 / 2.154 |
| 矩阵装配 | 3.734 / 4.414 | 3.613 / 3.977 |
| 线性求解 | 0.262 / 0.290 | 0.261 / 0.290 |
| 析构 | 0.091 / 0.110 | 0.076 / 0.101 |
| 核心总时间 | 7.509 / 8.507 | 6.977 / 7.821 |

总时间还包含状态复制、数值检查和状态更新等共同工作，不等于表内各阶段简单相加。求值和装配代码没有修改，其测量差异可能包含内存布局、缓存与运行波动；不能把全部核心时间收益归因于支撑查找本身。4 线程 D 进程峰值 RSS 从 132.7 MiB 到 124.1 MiB，这是整进程读数，包含输入快照等共同内存。

全部 10,100 个正式样本可用；C/D 各自优化前后的最终代价逐样本完全相同。101 个窗口的独立验证中，全后端 H 最大相对差 1.41e-15、g 为 2.35e-12；1/8 步 B/C 状态最大差 1.25e-12，A/D 为 2.88e-11。

原始结果：`build/p1_eee03_frozen/{all.csv,summary.json,manifest.json}`。持久摘要、全部阶段均值/P95、配对误差和运行来源见 [splbatch_p1_results.json](splbatch_p1_results.json)。

## 独立构造诊断

两版诊断程序分别使用 4 线程、stride 16、16 次重复，轮换全部采样 offset，各产生 1,616 个样本。每个窗口的全部观测恰好被采样一次；两版诊断另各通过 101 个窗口的 r/J、单步状态检查，最终代价逐样本完全相同。原始数据与命令在 `build/p1_diagnostics/`。

以下为扣除空探针及嵌套间隔后的**诊断估计**，单位 ms：

| 阶段 | 优化前 | P1 |
|---|---:|---:|
| Binding 构造 | 0.475 | 0.466 |
| 支撑识别 | 0.334 | 0.029 |
| 观测预计算 | 0.046 | 0.043 |
| 批次创建 | 0.079 | 0.079 |
| 原生 Jacobian 缓冲分配 | 0.067 | 0.065 |

该诊断支持主要收益来自支撑识别；Binding 构造仍约 0.47 ms，P2 尚未实施。不要将这些值当作无探针分项真值：包括观测存储、注册与清理等其余阶段后，校准合计为 1.353 / 0.912 ms，仍与正式构造均值 1.197 / 0.878 ms 不闭合。P1 也减少了探针作用范围，前后剩余扰动不同，不能将各分项差值相加充当正式收益。

## 完整 eee_03 回放

P1 后 D 在 4 线程、rate 1 下完成全包回放：1,806 帧、5,418 次单步优化，无不可用优化、空 LiDAR 约束帧或异常节点退出。逐帧验证三次 `优化一步 → 更新状态/去畸变 → associate`；扫描时间戳与 P1 前保存的完整 D 回放全部一致。

使用相同的 2,990 个 GT 时间点，补偿 Body→Prism 杆臂并做无尺度 SE(3) 对齐后，ATE RMSE 为 **0.026030 m**，P1 前保存的 D 结果为 **0.025977 m**。冻结输入前后代价完全相同；完整回放的轨迹结果保持接近。

本次回放的 D 构造均值 1.162 ms、核心均值 7.754 ms、整帧均值 77.352 ms / P95 89.879 ms。这里用于验证实际循环和轨迹；没有同轮交错重跑旧版完整回放，整帧读数不作为 P1 加速比例的证据。

原始记录：`build/p1_eee03_replay/replay_manifest.json`、`native_batch/{optimization.csv,frames.csv,trajectory/spline_log.csv}`；GT 结果：`build/p1_eee03_accuracy/accuracy.json`。回放与精度摘要也保存在下方 JSON 报告。

## 性能方法与复现

正式测试使用 101 个 eee_03 冻结窗口，1/4/8/16 线程、每配置 5 轮。每个独立进程先完整预热一遍，再测一次单步求解；交错运行 A、P1 前后 C、P1 前后 D，共 10,100 条正式样本。101 个窗口的全后端 H/g 和 1/8 步状态检查另进程执行，不并入计时。

修改前源码、程序与共享库保存在 `build/p1_before/`。脚本用各版本的 `LD_LIBRARY_PATH` 加载对应库，并保存 `ldd` 结果和 SHA-256；拒绝加载错误库或带构造探针的库。输入快照、源码、命令和结果摘要也保存在 manifest 中。正式计时期间不并行编译或回放数据。

```bash
cmake -S . -B build/noetic
cmake --build build/noetic --parallel 16 --target \
  slict_splbatch_ingestion_test slict_splbatch_incremental_test \
  slict_solver_comparison_test slict_solver_benchmark \
  slict_construction_control slict_construction_diagnostics slict_estimator
ctest --test-dir build/noetic --output-on-failure \
  -R '^slict_(solver_comparison|splbatch_ingestion|splbatch_incremental)$'
python3 scripts/benchmark_splbatch_p1.py \
  --before build/p1_before --snapshots build/abc_eee03_capture/snapshots \
  --output build/p1_eee03_frozen
```

输出目录必须尚不存在。重新运行使用新目录名；对照必须保留配套旧共享库，不能只复制可执行文件。

完整回放与 GT 检查：

```bash
source devel/setup.bash  # zsh 使用 devel/setup.zsh
/usr/bin/python3 scripts/run_abc_replay.py \
  --bag /home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag \
  --dataset ntuviral --output build/p1_eee03_replay \
  --backends native_batch --threads 4 --outer-iterations 3 \
  --duration 0 --rate 1 --no-snapshots --timeout 600
/usr/bin/python3 scripts/evaluate_ntuviral.py \
  --ground-truth build/abc_eee03_inputs/ground_truth.csv \
  --prism-calibration /home/jiadong/workspace/datasets/ntu/eee_03/leica_prism.yaml \
  --replay build/p1_eee03_replay --output build/p1_eee03_accuracy \
  --backends native_batch
```
