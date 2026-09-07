# 轻量 Bind：内联输入与直接支撑查找

本次实现用户确认的两项：输入元数据使用内联小数组；AddObservation 直接读取这些数组执行 P1 查找。保留现有 PrepareObservation 接口和 batch 数值求值，不增加 PrepareObservationViewImpl。

本文保留 P2 独立测量结果。后续并行 cost 创建实验已回退，当前恢复为 P1＋P2 串行建批；对照数据保留在 [历史实验结果](splbatch_parallel_creation_results.md)。

## 使用方式与存储

新增 `splbatch/inline_binding_input.hpp`：

```cpp
using Input = splbatch::InlineBindingInput<SplineContext, 16>;
Input input;
input.context.dt = dt;
input.AddParameter(rotation.data(), 4);
input.AddParameter(position.data(), 3);
channel.AddObservation(input, observation);
```

输入拥有自己的参数地址、尺寸和签名存储。默认内联容量为 16 个参数、4 个签名元素，超过容量后按几何增长回退到动态存储。内联表示不是借用 Bind 函数局部数组的裸 view；复制、返回和移动输入后仍有正确的存储所有权。输入只借用实际优化参数的地址，参数本体生命周期要求与旧接口相同。

SLICT 的新 `BindInput()` 使用 8 个 LiDAR 参数块或 10 个 IMU 参数块，签名为空，因此不需要为这些列表分配堆内存。它读取与原 Bind 相同的 `o.span` 来列出依赖，没有增加应用侧分组或区间判断。

C/D 的注册循环改为 `AddObservation(BindInput(...), o)`。旧 Bind、BatchBinding 的公开 vector 类型和 Key 保留；B 的 ScalarCost 继续使用旧拥有型 Binding。

## 支撑查找与所有权

P1 的 `SameBatchSupport` 现在允许左右两种输入类型。先核对地址、尺寸、签名的数组长度，再逐项比较全部有序内容；哈希直接遍历输入数组，不创建临时 BatchKey。

新支撑经过完整校验，随后通过 `ToBinding()` 创建原类型的拥有型 Binding，准备观测并成功追加后才发布批次。已有支撑直接使用索引，输入数组不复制进支撑表。batch 最终继续持有原类型 Binding，Ceres cost、Jacobian 布局及求值接口不变。

输入数组溢出时，AddParameter 先为地址和尺寸同时确保容量，再追加元素；分配失败可改变容量，但不会只追加半个参数。AddObservation 沿用 P1 的观测/loss scale 回滚及新批次发布规则。

## 不修改预计算接口的兼容方式

`BindingPreparationReuse<Evaluator>` 是新增的库内适配约定，默认返回 false，绝不自动根据 Context 类型或相等运算猜测可复用性。

- 旧拥有型输入：始终使用本次输入调用原 PrepareObservation，行为与 P1 一致。
- 新支撑的轻量输入：使用刚由本次输入创建的 Binding 准备观测。
- 已有支撑的轻量输入：适配明确确认准备语义兼容时，使用已存 Binding；否则用本次输入创建临时完整 Binding，再调用原 PrepareObservation。

SLICT 的两个 evaluator 明确适配为 `stored.context.dt == input.context.dt`：其预计算仅使用本次观测和 dt，且 prepared observation 按值保存观测及系数。dt 改变时走新 context 的回退，不修改已存 Binding 的 context；dt 恢复时仍可复用原 Binding。每条观测都会重新准备，未缓存 prepared observation。

该约定只作用于普通 channel 的新输入路径，`incremental_problem_batcher.hpp` 未修改。泛型 evaluator 不会自动获得无分配命中；它们可以继续使用旧输入，或在新输入上走安全回退。轻量输入的 prepared observation 不得保留指向临时输入元数据的引用，适配也不得依赖临时对象身份。

## 验证

三个 CTest 通过：原 A/B/C/D 数值与求解测试、普通 ingestion 测试、上游增量生命周期测试。新增内容包括：

- 实际 SLICT 8/10 参数输入与旧 Bind 的 Key 一致；同一 IMU batch 内 dt 改变、恢复的残差与逐条原生因子一致。
- 160 个支撑、480 条观测的有序/随机输入；逐条 context/loss scale、真正的完整哈希碰撞，以及混合表示的支撑比较。
- 内联边界 16、溢出边界 17、48 参数、长签名、空参数布局；复制、移动、赋值、清空后复制和移动后重用。
- 非法长度、空指针、重复参数、非法尺寸和 loss scale；Commit 后添加被拒绝。
- context 不兼容时刷新、原 context 恢复、未适配 evaluator 的安全回退，以及准备异常。
- 逐个注入地址/尺寸溢出分配、首次建批、命中追加、context 回退和索引创建的分配失败，验证恢复后的批次、残差和 loss scale。
- 容量足够时，从轻量输入构造到连续/表命中追加均没有 C++ new 调用。首次支撑、观测存储扩容、通用溢出和 context 回退仍可分配。

ingestion 新增用例通过 AddressSanitizer/UndefinedBehaviorSanitizer；沿用当前沙箱下关闭 LeakSanitizer 的设置，不将其算作泄漏检查。日志为 `build/p2_ctest.log`、`build/p2_sanitizers_test.log`。

## 测量与复现

P1 源码、程序及共享库已保存在 `build/p2_before/`。使用 101 个 eee_03 冻结窗口，与当前程序在独立进程中交错运行；明确加载各自共享库，正式程序没有构造探针。比较的是 P1 与 P1 加本次轻量输入，不混入旧 Bind 的 reserve 改造。

```bash
cmake --build build/noetic --parallel 16 --target \
  slict_splbatch_ingestion_test slict_splbatch_incremental_test \
  slict_solver_comparison_test slict_solver_benchmark \
  slict_construction_control slict_construction_diagnostics slict_estimator
ctest --test-dir build/noetic --output-on-failure \
  -R '^slict_(solver_comparison|splbatch_ingestion|splbatch_incremental)$'
python3 scripts/benchmark_splbatch_p1.py \
  --before build/p2_before --after-label p2 \
  --snapshots build/abc_eee03_capture/snapshots \
  --output build/p2_eee03_frozen
```

该脚本保留默认的 p1 标记以兼容此前命令；本次显式使用 p2 标记。复现时使用新的输出目录，避免覆盖已有记录。

## 正式冻结窗口结果

101 个窗口 × 5 轮 × 4 种线程数 × 5 个版本/后端组合，共 10,100 个正式样本。每组 505 个样本，单次优化一步；每个独立进程先完整预热一遍。下表单位为 ms，下降比例均相对本次重新测量的 P1。

| 后端 | 线程 | P1 构造 | 本次构造 | 构造下降 | P1 总耗时 | 本次总耗时 | 总耗时下降 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D：SPLBATCH + SLICT 原生优化 | 1 | 0.856 | 0.709 | 17.2% | 8.133 | 8.030 | 1.3% |
| D：SPLBATCH + SLICT 原生优化 | 4 | 0.888 | 0.736 | 17.2% | 6.905 | 6.704 | 2.9% |
| D：SPLBATCH + SLICT 原生优化 | 8 | 0.905 | 0.733 | 19.0% | 6.623 | 6.281 | 5.2% |
| D：SPLBATCH + SLICT 原生优化 | 16 | 0.970 | 0.799 | 17.6% | 6.767 | 6.521 | 3.6% |
| C：SPLBATCH + Ceres | 1 | 0.762 | 0.429 | 43.6% | 8.625 | 8.349 | 3.2% |
| C：SPLBATCH + Ceres | 4 | 0.976 | 0.645 | 33.9% | 5.144 | 4.885 | 5.0% |
| C：SPLBATCH + Ceres | 8 | 1.034 | 0.704 | 31.9% | 4.418 | 4.114 | 6.9% |
| C：SPLBATCH + Ceres | 16 | 0.996 | 0.662 | 33.5% | 4.299 | 3.979 | 7.4% |

所有 C/D 改动前后的对应窗口、轮次最终代价完全相同。独立的 101 窗口验证覆盖 A/B/C/D 的 H/g，以及 1/8 步求解：H 最大相对误差 `1.40977e-15`，g 最大相对误差 `2.34173e-12`；B/C 最大状态误差 `1.24083e-12`，A/D 为 `2.87176e-11`。验证运行不进入上述正式统计。

4 线程完整阶段均值（同一批原始 CSV，每列 505 个样本）：

| 阶段 | A 原生（ms） | D：P1（ms） | D：P1＋轻量输入（ms） |
|---|---:|---:|---:|
| 状态复制 | 0.000428 | 0.000518 | 0.000512 |
| 构造 | 0.000680 | 0.888012 | 0.735697 |
| 因子求值 | 3.715252 | 1.684492 | 1.731008 |
| H/g 组装（含稀疏化和阻尼） | 3.594168 | 3.573333 | 3.472928 |
| 线性求解 | 0.250973 | 0.255211 | 0.262337 |
| 状态更新 | 0.001156 | 0.001260 | 0.001298 |
| 末尾清理（batch、全局 J/r） | 0.016679 | 0.072543 | 0.069872 |
| 未单列耗时（按差值计算） | 0.423151 | 0.429886 | 0.430463 |
| 总耗时 | 8.002488 | 6.905255 | 6.704116 |

上一版对话表只列构造、求值、析构，却直接列出总耗时，漏掉了组装、线性求解等阶段，容易误认为前三行应该相加得到总耗时。此处补全。未单列耗时逐样本按 `total_ms - (reset_ms + build_ms + evaluate_ms + assemble_ms + linear_ms + update_ms + destroy_ms)` 计算；未舍入数据闭合，表中六位小数可能有末位舍入差异。

未单列耗时约 0.42–0.43 ms，绝大部分位于求解循环内。依据 `src/comparison/native.cpp` 的计时边界，该部分包含初始代价计算、全局 r/J 的有限值检查、循环内稀疏矩阵和求解器临时对象的析构，以及循环控制和计时边界开销；没有逐项探针，不能把差值全部归因于其中一项。`destroy_ms` 仅计末尾显式清理，不是整个 Solve 中所有对象的析构总和。

`solve_ms` 是包含求值、组装、线性求解、更新及循环内部未单列耗时的外层计时，不能再与这些子项相加。数值求值与组装代码未修改，其时间变化包含分配、缓存及运行波动，不能作为独立算法加速宣称。

4 线程 D 总耗时 P95 为 `7.773 → 7.554 ms`；16 线程均值虽改善，P95 为 `7.930 → 8.201 ms`，本次不能声称尾延迟改善。4 线程 D 的进程峰值 RSS 为 `125.6 → 132.4 MiB`，未观察到进程峰值内存下降。各组完整 P95、RSS 保存在 JSON。

本轮 16 线程 A 总耗时均值为 `6.743 ms`，D 为 `6.521 ms`，D 低约 3.3%；这是此组冻结窗口测量结果，不代表不同数据或在线整帧均能达到相同比例。

## 单线程下比较 A/D 的准备和求值成本

为排除 A 在观测循环内并行构造临时 factor 的影响，这里单独比较 `threads=1`：同一组 101 个冻结窗口、每配置 5 轮、每次优化一步，每列 505 个样本；数据来自已有正式无探针运行，并非新一轮测量。所有表中时间均为 ms。

| 阶段 | A 原生 | D：P1 | D：P1＋轻量输入 |
|---|---:|---:|---:|
| 状态复制 | 0.000440 | 0.000505 | 0.000508 |
| 构造 | 0.000262 | 0.856182 | 0.708615 |
| 求值 | 12.901258 | 2.951201 | 3.046395 |
| H/g 组装 | 3.612529 | 3.573213 | 3.503069 |
| 线性求解 | 0.252280 | 0.256366 | 0.262395 |
| 状态更新 | 0.001250 | 0.001207 | 0.001281 |
| 末尾清理 | 0.000152 | 0.072593 | 0.074971 |
| 未单列耗时（差值） | 0.434354 | 0.421681 | 0.432724 |
| 总耗时 | 17.202526 | 8.132950 | 8.029959 |

未舍入数据相加闭合，表中可能有末位舍入差异；未单列项的定义与上述 4 线程表相同。

A 的“构造＋求值＋末尾清理”为 12.902 ms，当前 D 为 3.830 ms，D 低 70.3%；总耗时由 17.203 ms 降至 8.030 ms，低 53.3%。因此，A 的独立构造列接近零不能解释为它的局部 Jacobian 分配和完整因子处理都很快：每观测临时 factor 的构造、分配、求值、释放仍计入 evaluate_ms，即使单线程也是如此。

这组数据确认单线程下 D 的整个因子处理流程更快，但没有把共享样条计算、参数复制、分配器行为等因素逐项隔离，不能把全部收益归因于其中一项。

相对本轮 P1，轻量输入使单线程 D 构造从 0.856 ms 降到 0.709 ms，总耗时从 8.133 ms 降到 8.030 ms（约 1.3%）；A 到 D 的总收益包含原有批处理实现，不能全部算作 P2 收益。

同轮当前 C 的单线程总耗时为 8.349 ms。C 使用 Ceres 求解器内部的计时和求值调用方式，不将它的 evaluate_ms 直接解释成与 A/D 一次 r/J 求值相同的工作量；本轮 P2 对照未包含 B。

## 独立构造诊断

正式运行结束后，另用带探针的程序运行：4 线程、stride 16、16 个轮换偏移、每版本 1,616 个样本。下表为扣除空探针开销后的阶段估计，单位 ms；它们用于判断成本来源，不能代替上面的正式时间。

| 阶段 | P1 估计 | 本次估计 |
|---|---:|---:|
| 输入 Binding / BindInput 构造 | 0.474 | 0.135 |
| 支撑识别 | 0.033 | 0.032 |
| 观测预计算 | 0.047 | 0.056 |
| cost 批次创建 | 0.077 | 0.078 |
| 原生 Jacobian 缓冲分配 | 0.061 | 0.058 |
| 首次支撑保存与观测追加等 | 0.146 | 0.149 |
| 原生注册其他开销 | 0.023 | 0.024 |
| 输入对象清理 | 0.084 | 0.041 |
| channel 清理 | 0.001 | 0.001 |

输入构造估计从 `0.474` 降至 `0.135 ms`，与消除逐观测列表堆分配的预期方向一致。新支撑的 `ToBinding()` 在 AddObservation 内执行，仍属于正式 build_ms；诊断中归入首次支撑保存/观测追加阶段，不在 BindInput 工厂阶段。context 不兼容时的转换归入预计算阶段。

诊断估计之和为 `0.947 → 0.573 ms`，正式构造是 `0.888 → 0.736 ms`，并不闭合。空探针校准不能消除代码生成、缓存、分配器和采样扰动；不能用诊断中 Binding 减少的 `0.339 ms` 推算 D 的正式收益（实测减少 `0.152 ms`），也不将各阶段强制归一化为“精确占比”。两版诊断的 r/J/状态验证均通过，1,616 对最终代价相同。

正式原始数据与哈希：`build/p2_eee03_frozen/{all.csv,summary.json,manifest.json}`；诊断：`build/p2_diagnostics/`。保留的 P1 程序与源码哈希：`build/p2_before/sha256.json`。

## 完整 eee_03 回放

使用新 estimator 和正式共享库，4 线程、1 倍速、关闭快照导出，完成全部 1,806 帧、5,418 次单步优化。每帧均为三次“优化一步 → 更新状态、去畸变 → associate”，无求解失败、无空 LiDAR 帧，未超时；最大队列为 1 个包。

独立逐条比较本次与 P1 的 frames.csv，1,806 个扫描结束时间戳完全一致。GT 使用相同的 2,990 个时间点，时间范围和逐条时间戳均一致。按原有棱镜偏移和刚性 SE(3) 对齐评估，ATE RMSE 为 **0.026000 m**；P1 参考为 0.026030 m。轨迹没有逐位相同，因此这里只报告精度接近；冻结窗口的前后最终代价仍完全相同。

本次在线单次优化构造均值为 0.919 ms、总耗时均值 7.226 ms；整帧均值 75.754 ms、P95 86.200 ms。在线观测和窗口随运行变化，这些数字不与冻结窗口混用；本次完整回放主要验证覆盖、循环节奏和精度，没有同轮重复运行 P1 来判定整帧加速比例。

回放记录：`build/p2_eee03_replay/replay_manifest.json`；精度与轨迹图：`build/p2_eee03_accuracy/`。正式结果、诊断、验证和哈希汇总保存在 [机器可读结果](splbatch_light_binding_results.json)。

```bash
source devel/setup.zsh
/usr/bin/python3 scripts/run_abc_replay.py \
  --bag /home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag \
  --dataset ntuviral --output build/p2_eee03_replay \
  --backends native_batch --threads 4 --outer-iterations 3 \
  --duration 0 --rate 1 --no-snapshots --timeout 600
/usr/bin/python3 scripts/evaluate_ntuviral.py \
  --ground-truth build/abc_eee03_inputs/ground_truth.csv \
  --prism-calibration /home/jiadong/workspace/datasets/ntu/eee_03/leica_prism.yaml \
  --replay build/p2_eee03_replay --output build/p2_eee03_accuracy \
  --backends native_batch
```
