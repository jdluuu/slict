# native_batch 构造耗时诊断

2026-09-07，完成优化方案 P0。这里只增加诊断并测量，尚未实施 Binding 缓存、支撑识别优化或缓冲池。

## 结果

同一组 101 个 eee_03 冻结窗口，D 每次优化一步、求值线程数为 4。构造本身串行。正常构建测了 16 轮、1616 个样本：**构造均值 1.164 ms，P95 1.318 ms**。这是本次正式总时间；此前的 1.321 ms 是历史运行的均值，未强行把本次分项缩放到历史数值。

下面是 1/16 抽样、轮换全部 16 个采样偏移、扣除空探针估计成本后的分项。百分比以**校准后已归因工作合计 1.264 ms**为分母，不是无探针的 1.164 ms。最后一列是 1、1/4、1/16 三种计时配置之间的范围，不是统计置信区间。

| 分项 | 校准估计 ms/窗口 | 已归因工作占比 | 三种配置的估计范围 ms |
| --- | ---: | ---: | ---: |
| Binding 构造 | 0.471 | 37.3% | 0.444–0.473 |
| 支撑识别 | 0.321 | 25.4% | 0.320–0.329 |
| 观测预计算 | 0.046 | 3.7% | 0.046–0.050 |
| 批次 cost 创建及写入计划 | 0.073 | 5.8% | 0.073–0.077 |
| 原生 Jacobian 缓冲分配 | 0.055 | 4.3% | 0.055–0.059 |
| 观测存储与添加杂项 | 0.193 | 15.2% | 0.184–0.193 |
| 原生参数注册，不含 Jacobian 缓冲分配 | 0.022 | 1.7% | 0.022–0.023 |
| 临时 Binding 释放及调用边界杂项 | 0.081 | 6.4% | 0.074–0.081 |
| channel 提交后清理 | 0.002 | 0.2% | 0.002–0.002 |

**Binding 构造和支撑识别合计约占已归因工作的 63%。** 三种采样配置的排序一致。观测预计算与 Jacobian 缓冲分配都只有约 4%，因此不应优先重写 evaluator 或把分配全部归因于原生 Jacobian 缓冲。

Binding 的逐观测动态数组构造/释放，以及 SPLBATCH 的逐观测 key、校验、查表和观测容器处理，是接下来应优先处理的部分。库内支撑识别优化 P1 和低成本 Binding 构造 P2 比单独合并原生 Jacobian 缓冲 P3 更值得先验证；用户仍正常逐条 AddObservation，分组职责留在 SPLBATCH 内部。

## 探针扰动与精度边界

| 构建/运行模式 | 正式样本数 | 构造均值 ms | 相对相同轮次无探针对照 |
| --- | ---: | ---: | ---: |
| 正常构建，无诊断代码 | 1616 | 1.164 | — |
| 诊断构建，但关闭采集 | 505 | 1.240 | +7.4% |
| 全量逐观测计时 | 505 | 2.926 | +153.5% |
| 1/4 观测抽样 | 404 | 1.717 | +48.4% |
| 1/16 观测抽样 | 1616 | 1.393 | +19.6% |

以上百分比使用各组相同轮次的对照，不是直接用汇总表第一行相除。共 4646 个计时样本，另有独立验证运行。每个进程先遍历所有窗口预热一次；进程顺序轮换，不并行运行正式计时。

空叶子计时区间平均约 16.1 ns，完整空探针的外部开销约 35.2 ns；嵌套子区间还会给父区间留下约 19.1 ns 的计时/记账间隙。逐观测有多个区间，数千次调用累计后不可忽略。

原始时长、采样次数、嵌套次数、外推值和校准值都保留在 CSV/JSON 中。校准使用：

```text
阶段校准估计 = 抽样外推的独占时间
             - 空叶子区间成本 × 外推区间次数
             - 空嵌套间隙成本 × 外推直接子区间次数
```

三种采样方式的校准合计分别为 1.246、1.256、1.264 ms，比各自正常对照高约 0.09–0.10 ms。**空探针扣除无法完全消除编译布局、分支、分配和缓存扰动；这些分项是诊断估计，不能宣称已经精确闭合到正式总时间。** 没有把误差塞进“其他”项、截断负余额或强制缩放毫秒值来制造闭合。约 8% 的残余差异足以影响精细百分比，但三种配置共同支持主要成本的排序。

另尝试了无插桩 gprofng 调用栈采样。4 线程运行构造均值为 1.149 ms，单线程为 1.146 ms，但主线程的构造调用栈样本太少：两次采样仅归到约 47/35 个 1 ms 构造 tick。单线程进程使用了 55.48 s 用户 CPU，采样报告只计到 0.553 s。故这些调用栈结果未用于分项占比，也未作为独立验证占比的证据。

## 分项口径

- **Binding 构造**：当前 `Bind()` 返回参数指针/尺寸数组及 Context；包括其构造中的扩容，不含完整临时 Binding 在 AddObservation 返回后的释放。
- **支撑识别**：`binding.Key()` 中的参数校验和 key 复制、哈希查找、命中后的兼容性比较，以及新支撑的表插入。
- **观测预计算**：仅 `evaluator_->PrepareObservation(binding, observation)`，包括返回 PreparedObservation 的构造；不含追加到观测 vector。
- **批次创建**：Commit 内参数列表复制、cost 对象创建及其 Initialize，包括写入计划；不含注册回调。
- **原生 Jacobian 缓冲分配**：每个未固定参数块的 Eigen 缓冲 resize。该调用不代表后续首次写入和物理页面触碰的全部成本，后者仍在求值阶段。
- **观测存储与添加杂项**：AddObservation 的剩余独占时间，包括 loss scale 检查、pending Binding 保存、观测与 loss scale 容器扩容/追加及临时 key 释放。
- **原生参数注册**：注册回调的剩余独占时间，包括参数查表、元数据数组管理和 Entry 保存，已扣除 Jacobian 缓冲区间。
- **临时 Binding 释放及调用边界杂项**：外层添加表达式中，扣除 Bind 与 AddObservation 后的剩余时间。
- **channel 提交后清理**：清理支撑表、pending 批次及 sealed 标志。

建批前的全局 r/J 分配、原生参数映射初始化和部分函数/对象边界工作没有归到上述九项。全量诊断能记录其原始未归因余额，但其中还包含探针记账成本，不能作为精确原生 setup 时间。

每个窗口平均有 7691.4 条观测、61.25 个观测批次及 551.27 次未固定参数块的 Jacobian resize。支撑识别分成两个计时区间，因此其区间次数约为观测数的两倍，不是执行了两次完整判批。

## 实现与检查

- [construction_diagnostics.hpp](../third_party/splbatch/include/splbatch/construction_diagnostics.hpp)：编译期开关、嵌套独占计时、按观测采样、原始及外推计数。
- [problem_batcher.hpp](../third_party/splbatch/include/splbatch/problem_batcher.hpp) 与 [native_batch.cpp](../src/comparison/native_batch.cpp)：仅增加诊断区间。
- [construction_benchmark.cpp](../src/comparison/construction_benchmark.cpp)：同一程序源码构建无探针和诊断两个可执行文件；均调用完整的单步 Solve，输入 I/O、验证和校准不在 build_ms 内。
- [运行脚本](../scripts/profile_native_batch_construction.py)：独立进程轮换、覆盖全部采样偏移、结果一致性检查和汇总。

正常库与诊断库分别链接到不同可执行文件，避免模板的不同宏配置混入同一进程。诊断目标是 EXCLUDE_FROM_ALL，默认构建不会启用探针。

`cmake --build ... --parallel 16` 构建通过。正常库 `.text` 的 SHA-256 与诊断改动前完全相同：`8447c9eab303a80f8ff8f00c19a423807655a4a91cc14869b23e80e8fd187c72`，且不含诊断符号。SLICT evaluator 和批次计算头文件的 SHA-256 未变。

原 `slict_solver_comparison` CTest 通过。另对 101 个窗口开启全量探针验证：最大 r 相对差 `5.05e-12`、J 相对差 `1.75e-14`、单步状态差 `1.50e-12`。全部 4646 个计时样本的最终代价与对应正常对照完全相同。1/4 和 1/16 的全部偏移分别覆盖每条观测恰好一次；批次数、预处理次数和嵌套计时边界检查通过。

## 复现与原始数据

```bash
cmake -S . -B build/noetic
cmake --build build/noetic --parallel 16 --target \
  slict_construction_control slict_construction_diagnostics \
  slict_solver_comparison_test
ctest --test-dir build/noetic --output-on-failure -R '^slict_solver_comparison$'
/usr/bin/python3 scripts/profile_native_batch_construction.py \
  --snapshots build/abc_eee03_capture/snapshots \
  --output build/construction_profile_eee03 \
  --threads 4 --max-snapshots 101 --validate 101
```

复跑时使用新的输出目录。此命令复用本机已配置的 Noetic/Ceres 依赖，见 [构建记录](build_noetic.md)。

- [持久化结果、环境、代码及输入校验和](native_batch_construction_profile.json)
- [原始全部计时 CSV](../build/construction_profile_eee03/all.csv)
- [各阶段原始值、校准值及逐轮估计](../build/construction_profile_eee03/summary.json)
- [进程命令和二进制/共享库记录](../build/construction_profile_eee03/manifest.json)
- [101 窗口数值验证日志](../build/construction_profile_eee03/validation_r-1.log)
- [正常库机器码一致性检查](../build/construction_profile_binary_check/check.json)
