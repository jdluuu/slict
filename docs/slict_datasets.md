# SLICT 论文数据集与 A/B/C 测试入口

当前 `noetic` 是 SLICT2 的 ROS 1 版本；SLICT1 对应 `slict.1.0`，两篇论文的数据集和传感器选择应分开看。[官方仓库说明](https://github.com/brytsknguyen/slict)

| 论文 | 实际评测数据 |
| --- | --- |
| SLICT，RA-L 2023 | NTU VIRAL、Newer College、作者自采 ATV 数据。NTU 实验合并水平和垂直两台雷达。[实验 IV-A～C](https://arxiv.org/html/2211.03900v2#S4) |
| SLICT2 / *Eigen Is All You Need*，RA-L 2024 | NTU VIRAL、MCD。NTU 使用水平 16 线雷达和高频 IMU，MCD 使用 Livox 数据。[实验 IV-A～B](https://arxiv.org/html/2402.02337v2#S4) |

SLICT2 的 NTU 表 I 包含 `eee`、`nya`、`sbs`、`rtp`、`tnp`、`spms` 六组，每组 `01/02/03`，共 18 个序列。论文参数为四阶样条、0.01 s 节点间隔、三帧窗口、三轮内部更新、每轮重新关联两帧、最多 8000 个 LiDAR 因子。当前 ABC 默认采用这些调度参数，但使用共同边缘化模型，不能直接当作未经修改的论文精度复现。[SLICT2 实验设置与表 I](https://arxiv.org/html/2402.02337v2#S4)

## 建议先下载 NTU VIRAL

先用较小的 `eee_03` 验证整条流程，再加室内 `nya_01` 和另一室外场景 `sbs_01`。以下大小和时长来自 [官方完整下载表](https://ntu-aris.github.io/ntu_viral_dataset/#downloads)，页面也提供备用镜像及真值 CSV。

| 序列 | 下载大小 | 时长 | 建议用途 |
| --- | ---: | ---: | --- |
| `eee_03` | 4.3 GB | 181.4 s | 第一条完整 A/B/C 基线 |
| `nya_01` | 8.6 GB | 396.3 s | 室内场景补充 |
| `sbs_01` | 7.8 GB | 354.2 s | 室外场景补充 |

`tnp_02` 可作为后续困难场景单独报告失败率；SLICT2 论文专门讨论了其玻璃、日照及高度约束不足的问题。[论文图 5](https://arxiv.org/html/2402.02337v2#S4)

下载 ZIP 并解压后，把实际 `.bag` 路径传给已实现的入口：

```bash
source devel/setup.bash  # zsh 使用 devel/setup.zsh
/usr/bin/python3 scripts/run_abc_replay.py \
  --dataset ntuviral --bag /path/to/eee_03.bag \
  --output build/abc_eee03_repeat \
  --duration 0 --rate 0.5 --threads 4 --outer-iterations 3 \
  --timeout 1800 --no-snapshots
```

三组的内部迭代数固定为 1。该命令用于整帧耗时和跟踪检查；若要采集离线快照，在另一个输出目录去掉 `--no-snapshots`。输出目录必须是新目录。

[NTU 配置](../config/abc_ntuviral.yaml) 只订阅 `/os1_cloud_node1/points` 与 `/imu/imu`，保留原库的标定及扫描结束时间戳约定；不要用 R3LIVE 的外参，也不要把原双雷达配置直接当作 SLICT2 论文的单雷达配置。统一入口是 [run_abc.launch](../launch/run_abc.launch)，更多选项见 [对比使用文档](solver_comparison.md)。

真值包含在原 bag，也可从 [官方真值仓库](https://github.com/ntu-aris/ntuviral_gt) 获取。计算 ATE 前应按 [官方评估教程](https://ntu-aris.github.io/ntu_viral_dataset/evaluation_tutorial.html) 将估计转换到棱镜测量点、匹配时间并对齐坐标系；棱镜与 IMU 相距约 0.4 m，直接比较会引入假误差。

已增加 [评估脚本](../scripts/evaluate_ntuviral.py)，在同一批真值 header 时间戳上从控制点求实际样条位姿，读取包内 `leica_prism.yaml` 补偿棱镜偏移，再分别做无尺度的 SE(3) 对齐。没有拟合时间偏移，也不会把控制点直接当作采样轨迹。完成三组回放后可运行：

```bash
/usr/bin/python3 scripts/evaluate_ntuviral.py \
  --bag /path/to/eee_03/eee_03.bag \
  --prism-calibration /path/to/eee_03/leica_prism.yaml \
  --replay build/abc_eee03_repeat --output build/abc_eee03_repeat_accuracy
```

输出包括 `accuracy.json`、逐时刻误差 CSV、采样位姿 TUM 文件和轨迹/误差图。脚本依赖 NumPy、SciPy 和 Matplotlib；若直接读取 bag，还需 ROS Python 环境。三组必须处理相同帧时间戳，评估才会继续。`eee_03` 的真值只覆盖后面的飞行段，报告会同时给出共同样本数与真值覆盖率；回放处理完整性仍由回放日志检查。

本机已用 `/home/jiadong/workspace/datasets/ntu/eee_03/eee_03.bag` 完成三组全序列测试，正式回放位于 `build/abc_eee03_single_step`，评估位于 `build/abc_eee03_accuracy`。耗时、真值精度和复现命令见 [eee_03 实测报告](solver_comparison_eee03.md)。

## MCD 和 Newer College

[MCD 官方下载页](https://mcdviral.github.io/Download.html) 将数据按传感器拆成不同 bag。SLICT2 表 II 的六条 `xxx_day_01/02/10`、`xxx_night_04/08/13`，按编号与下载表对应为 `ntu_day_01/02/10`、`ntu_night_04/08/13`；这是根据两张表的编号对应关系作出的判断。论文耗时分析使用 `xxx_day_01`。[SLICT2 表 II 和 IV-B](https://arxiv.org/html/2402.02337v2#S4)

MCD 可只下载 Livox、VN100 IMU、真值和 ATV 标定：例如下载页列出 `ntu_day_01` 的 Livox 为 542 MB、VN100 为 32 MB；`ntu_day_02` 分别为 196 MB 和 12 MB。无需为了 LIO 下载全部相机数据。当前 ABC 一键入口仅适配 NTU VIRAL 与 R3LIVE；MCD 需要单独接入其分包回放、话题和外参后才能测量。[MCD 下载表与标定](https://mcdviral.github.io/Download.html)

[Newer College 官方入口](https://ori-drs.github.io/newer-college-dataset/) 适合补充 SLICT1 的评测；它不是 SLICT2 这篇论文表 I/II 的测试集。仓库存在其他数据集 launch 并不表示这些数据都在对应论文中跑过。
