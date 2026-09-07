# 已分组 batch 并行创建实验

**状态：已按用户要求回退。** 当前代码恢复为实验前的 P1＋P2 串行实现；并行 API、CLI/Options 开关、CSV 扩展、专项测试和测量脚本均已移除。下文保留实验时的实现、命令和结果，所列并行命令不适用于当前代码。实验源码改动和脚本归档在 `build/parallel_creation_archived_source/`，原始计时数据保持不变。 回退后已以 `--parallel 16` 重编译 estimator、基准和诊断程序；原有三组 CTest 通过，共享库与基准可执行文件的 SHA-256 均与实验前保存版本完全一致。

本实验比较当前 P1＋轻量输入的串行建批，与“已分组 batch 的 cost 创建并行”。默认仍串行，不自动启用。没有同时实施 P3 的统一 Jacobian 缓冲或 P4 的跨轮工作区复用。

## 实现范围

普通 ProblemBatcher 新增显式 `CommitParallel(threads)`。每个 channel 为其已分组 batch 预分配结果槽位，然后用 OpenMP `parallel for schedule(static)` 创建各自的 cost；参数指针复制、Binding/观测存储转移、cost 内部布局校验和 Jacobian 写入计划创建在工作线程中执行。

所有线程结束后，先检查创建异常，再按首次支撑出现顺序串行调用 registry。IMU 和 LiDAR channel 仍依次提交，所以多线程模式通常有两个并行区；不跨 channel 合并任务。空 channel、单 batch 和 threads=1 走原串行路径。

以下工作保持原样：逐条 BindInput/AddObservation、P1 支撑识别、观测 PrepareObservation、SLICT 原生参数映射、registry 注册、局部 Jacobian 缓冲分配、batch 数值求值、全局 J scatter、H/g 组装和求解。IncrementalProblemBatcher 未修改。

并行是显式选择：调用者需确保 cost 构造、evaluator 的布局查询、Context 移动及 loss policy 复制可并发执行；不能仅依据 const 接口推断线程安全。本次 SLICT evaluator 的相关操作符合条件。AddObservation 与 Commit 仍不能重叠调用。

工作线程只修改自己的 pending batch 和结果槽位，不操作 registry。工作线程异常被捕获，等待所有线程结束后重新抛出；此 channel 的 cost 全部创建成功之前不发布残差。开始消费 pending 数据后如失败，channel 进入终止状态，后续 Commit 重抛原异常、AddObservation 被拒绝，不从已移动的数据重试。registry 自身注册失败时可能已经接收前面的 cost，现有 registry 接口没有事务回滚，本实现不承诺整个 ProblemBatcher 的原子提交。

## 开关与计时

SLICT Options 增加 `batch_creation_threads`，默认 1；冻结窗口程序新增 `--batch-creation-threads N`。C/D 均支持该开关，A/B 行为不变。`--threads N` 继续只指定原有求值/求解线程数。CSV 末尾附加 `batch_creation_threads`，记录实际请求的创建线程数。

两个线程数独立设置。例如求值保持 8 线程，对比：

```bash
devel/lib/slict/slict_solver_benchmark \
  --snapshots build/abc_eee03_capture/snapshots --backend native_batch \
  --threads 8 --batch-creation-threads 1 --iterations 1 \
  --repeat 5 --warmup 1 --validate 0 --output build/creation_serial.csv
devel/lib/slict/slict_solver_benchmark \
  --snapshots build/abc_eee03_capture/snapshots --backend native_batch \
  --threads 8 --batch-creation-threads 8 --iterations 1 \
  --repeat 5 --warmup 1 --validate 0 --output build/creation_parallel.csv
```

正式实验使用下面的轮换脚本，而非只执行上面两个连续命令。所有暂存分配、线程启动/唤醒、同步、串行注册及建批临时对象清理都留在 build_ms 内；最终 cost 清理仍在 destroy_ms。使用无构造探针的正式程序；现有分项诊断面向串行路径，本实验不将其输出解释为并行阶段计时。

## 验证与复现

四组 CTest 通过。新增测试实际记录参与 cost 布局创建的线程，确认线程数大于 1 且 batch 足够时有多个线程工作，并确认 registry 始终在并行区之外调用。

覆盖 1/2/4/8/16 创建线程、0/1/31/64 个 batch、逆序支撑输入、批内观测顺序和 loss scale；注入布局查询异常、bad_alloc 和串行注册异常，检查创建失败不发布、失败后重试不重复注册。实际 C/D 在固定控制点、仅 IMU、仅 LiDAR、混合观测条件下，串并行 r/J 对照及 8 步状态对照通过。原 ingestion 和增量回归测试继续通过。

并行创建专项测试通过 ASan/UBSan，LeakSanitizer 沿用沙箱兼容设置关闭；未将此算作 ThreadSanitizer 或泄漏检查。日志：`build/parallel_creation_ctest.log`、`build/parallel_creation_sanitizers/test.log`。

原串行源码、共享库、程序及哈希保存在 `build/parallel_creation_before/`。正式实验同时测量：旧程序 before、新程序 serial、新程序 parallel，以区分源码改造本身的影响和并行机制的影响。

```bash
cmake --build build/noetic --parallel 16 --target \
  slict_solver_benchmark slict_solver_comparison_test \
  slict_splbatch_ingestion_test slict_splbatch_incremental_test \
  slict_splbatch_parallel_creation_test
ctest --test-dir build/noetic --output-on-failure \
  -R '^slict_(solver_comparison|splbatch_ingestion|splbatch_incremental|splbatch_parallel_creation)$'
python3 scripts/benchmark_parallel_creation.py \
  --before build/parallel_creation_before \
  --snapshots build/abc_eee03_capture/snapshots \
  --output build/parallel_creation_frozen
```

每个进程明确加载自己的共享库，记录二进制、源码、输入和 CSV 哈希。独立进程完整预热后，C/D、各模式按轮次交错运行。1 线程只测 before/serial，2/4/8/16 线程测全部三种模式；每配置 101 个相同窗口、5 轮、每次优化一步。性能测试不与编译或其他测量重叠。复现时使用新的输出目录。

## 正式结果

**本版实验没有获得稳定的整体加速，不建议据此启用并行创建或用本版直接替换原串行实现。** 此结论限定于当前“每个 channel 单独并行创建 cost、最后串行注册”的实现和本组冻结窗口，不能推广为所有并行建批方案无效。

原实现指本轮改动前的 P1＋轻量输入串行 C/D，不是 SLICT 原生 A。正式数据共 14,140 个样本，每个线程数/后端/模式组合 505 个样本。下面正的总耗时变化表示变慢，负值表示变快。单位 ms。

| 后端 | 求值/并行创建线程 | 原实现构造 | 并行版构造 | 原实现总耗时 | 并行版总耗时 | 总耗时变化 |
|---|---:|---:|---:|---:|---:|---:|
| D | 2 | 0.683 | 0.711 | 7.729 | 7.861 | +1.7% |
| D | 4 | 0.718 | 0.728 | 6.700 | 6.827 | +1.9% |
| D | 8 | 0.709 | 0.716 | 6.183 | 6.492 | +5.0% |
| D | 16 | 0.798 | 0.827 | 6.279 | 6.792 | +8.2% |
| C | 2 | 0.564 | 0.581 | 6.199 | 6.186 | -0.2% |
| C | 4 | 0.612 | 0.645 | 4.732 | 4.871 | +2.9% |
| C | 8 | 0.672 | 0.661 | 3.982 | 4.065 | +2.1% |
| C | 16 | 0.660 | 0.691 | 3.883 | 4.578 | +17.9% |

C 的 2 线程总耗时仅改善约 0.2%，各轮正负变化都有，不能作为稳定收益。D 的 8/16 线程轮间波动也较明显：例如 16 线程五轮总耗时变化约 -7.1%、+17.2%、+14.3%、+14.0%、+4.9%；表中为所有 505 个样本的均值，不是选择某轮。完整逐轮变化在 JSON。

## 区分代码改造与并行开关

新代码 threads=1 的提交仍走原先逐 cost 创建并立即注册的循环，不使用并行结果槽位。但是新增分支、对象布局以及重新编译后的代码会影响性能，因此保留新代码 serial 对照，不能把 before→parallel 的全部差异归因于 OpenMP。下面均在同一个新程序内比较。

| 后端 | 求值线程 | serial 构造 | parallel 构造 | serial 总耗时 | parallel 总耗时 | 总耗时变化 |
|---|---:|---:|---:|---:|---:|---:|
| D | 2 | 0.732 | 0.711 | 7.917 | 7.861 | -0.7% |
| D | 4 | 0.761 | 0.728 | 6.923 | 6.827 | -1.4% |
| D | 8 | 0.770 | 0.716 | 6.635 | 6.492 | -2.1% |
| D | 16 | 0.825 | 0.827 | 6.604 | 6.792 | +2.8% |
| C | 2 | 0.598 | 0.581 | 6.211 | 6.186 | -0.4% |
| C | 4 | 0.657 | 0.645 | 4.818 | 4.871 | +1.1% |
| C | 8 | 0.717 | 0.661 | 4.018 | 4.065 | +1.2% |
| C | 16 | 0.699 | 0.691 | 3.984 | 4.578 | +14.9% |

原实现→新 serial 的对照也必须保留：

| 后端 | 求值线程 | 原实现构造 | 新 serial 构造 | 原实现总耗时 | 新 serial 总耗时 |
|---|---:|---:|---:|---:|---:|
| D | 1 | 0.675 | 0.722 | 7.767 | 7.928 |
| D | 2 | 0.683 | 0.732 | 7.729 | 7.917 |
| D | 4 | 0.718 | 0.761 | 6.700 | 6.923 |
| D | 8 | 0.709 | 0.770 | 6.183 | 6.635 |
| D | 16 | 0.798 | 0.825 | 6.279 | 6.604 |
| C | 1 | 0.415 | 0.434 | 8.252 | 8.301 |
| C | 2 | 0.564 | 0.598 | 6.199 | 6.211 |
| C | 4 | 0.612 | 0.657 | 4.732 | 4.818 |
| C | 8 | 0.672 | 0.717 | 3.982 | 4.018 |
| C | 16 | 0.660 | 0.699 | 3.883 | 3.984 |

新 serial 同样出现性能退化，当前数据没有进一步隔离编译布局、分配地址/尺寸类别与其他因素的贡献。因此该补丁保留为实验开关，默认仍串行，但不能宣称新 serial 与原版等速。若继续推进，应先消除默认串行路径的退化，再评估并行创建的净收益。

## 清理和总时间的影响

并行创建结束不意味着成本结束。以下为末尾 destroy_ms 与 total_ms 的 P95，均包括此前正式测试中的全部样本。

| 后端 | 线程 | 原实现末尾清理均值 | 并行版末尾清理均值 | 原实现 total P95 | 并行版 total P95 |
|---|---:|---:|---:|---:|---:|
| D | 2 | 0.068 | 0.070 | 8.526 | 8.692 |
| D | 4 | 0.067 | 0.075 | 7.476 | 7.707 |
| D | 8 | 0.062 | 0.065 | 7.223 | 7.794 |
| D | 16 | 0.080 | 0.094 | 7.438 | 8.121 |
| C | 2 | 0.069 | 0.068 | 7.127 | 7.147 |
| C | 4 | 0.076 | 0.078 | 5.522 | 5.702 |
| C | 8 | 0.120 | 0.121 | 4.587 | 4.706 |
| C | 16 | 0.190 | 0.237 | 4.410 | 5.236 |

## 结果解释与边界

本组窗口平均约 62.24 个残差块（包含先验），IMU/LiDAR 分成两个 channel，创建任务粒度较小。此前串行诊断中 cost 创建约 0.078 ms，仅占构造的一部分；这不能作为本轮并行阶段的精确计时，但说明仅并行这一阶段的潜在收益有限。

本次并行路径新增结果暂存和两次并行区同步，并改变了“逐个创建后立即注册”的分配顺序。cost 在工作线程分配，通常在主线程销毁，也可能影响分配器和缓存行为；C 还会随后使用自身求解线程。这些都是可能影响总耗时的机制，本轮未进一步隔离线程等待策略、分配器或缓存行为，不能指定某一项为全部退化的原因。

观察到 D 的新程序内部在 2/4/8 线程切换并行后，构造均值下降约 2.9%/4.3%/7.0%，但不足以超过原版；C 的 8 线程构造虽下降，总耗时仍变慢，16 线程求值和清理也受到影响。因此不能只报告 build_ms 中的一点下降就认定优化成功。

下一步更适合先恢复 serial 基线性能，再分别验证集中缓冲分配，或让更多观测准备工作进入并行范围。不能从这次实验推断 P3/P4 的实际收益。

101 个窗口的独立验证覆盖 A/B/C/D 的 H/g 和 1/8 步状态：H 最大相对误差 1.40977e-15，g 为 2.34173e-12；B/C 最大状态误差 1.24083e-12，A/D 为 2.87176e-11。所有正式样本的 before/serial/parallel 配对最终代价完全相同，所有求解 usable。未做新的在线整包回放，本实验用于冻结窗口效率对比，默认在线路径仍使用串行创建。

原始 CSV、逐轮统计、加载路径及哈希：`build/parallel_creation_frozen/`。可持久查看的汇总见 [JSON 结果](splbatch_parallel_creation_results.json)。
