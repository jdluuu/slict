# 本机 Noetic 构建记录

2026-09-06，`noetic` / `0a79e9756c532cb2ba93990ff15edfc059367c58` 的 12 个原始可执行目标全部编译成功。以下记录首次基线构建；后续新增的 A/B/C 后端、测试、回放及源码改动见 [实现文档](solver_comparison.md)。`.gitignore` 排除本地 `build/`、`devel/`。

本机环境是 Ubuntu 24.04.2、ROS Noetic、GCC 13.3、Eigen 3.4、SuiteSparse 7.6.1。所有构建命令均使用 `--parallel 16` 或 `-j16`。主项目保持原 CMake 的 Release 设置（`-O3 -Wall -g`），日志中仍有原有代码警告；首次基线构建阶段没有执行 SLAM 数据回放。

在仓库根目录重新编译、加载环境：

```zsh
cmake --build build/noetic --parallel 16
source devel/setup.zsh
```

Bash 用户使用 `source devel/setup.bash`。可执行文件位于 `devel/lib/slict/`，包括 estimator、sensorsync、imu_odom、relocalization 和 8 个点云格式转换器。已检查全部 12 个文件的动态库解析，并验证 `rospack find slict` 指向当前仓库。

## 本地依赖处理

依赖全部位于 `build/deps/`，没有替换系统安装：

- **Ceres 2.1.0**：系统 Ceres 2.2 不再提供本标签需要的 LocalParameterization 接口。源码提交 `f68321e7de8929fbcdb95dd42877531e64f72f66`，安装前缀 `build/deps/install`，启用 SuiteSparse，关闭可选 CXSparse/CUDA。为兼容 SuiteSparse 7，放宽 CMake 版本宏的空格匹配，并对协方差 QR 调用的列数显式转换为 `SuiteSparse_long`。补丁保存在 `build/ceres21-suitesparse7.patch`。
- **UFOMap**：上游 `devel_surfel` 最新提交已迁到 ROS2，使用其 ROS1 提交 `1acb0e6a2ba8748dba44229e5541eea199e10b32`。核心库装入同一本地前缀；`ufomap_msgs` 和 `ufomap_ros` 在 `build/deps/ros_ws` 编译。
- **Livox 消息接口**：SLICT 只消费 `CustomMsg`、`CustomPoint`。driver1 使用本机 ClampSLAM 中的原消息定义，driver2 使用官方仓库提交 `4a1def929e5b59c7a8122d19fce6efba581ce9f7` 的原消息定义，通过本地 catkin 消息包生成头文件。没有构建硬件采集驱动或安装 Livox SDK；两个 SLICT Livox 转换器均已编译。

若仅需重新生成主项目 CMake 配置，保留上述依赖目录后执行：

```sh
cmake -S . -B build/noetic \
  -DCATKIN_DEVEL_PREFIX="$PWD/devel" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCeres_DIR="$PWD/build/deps/install/lib/cmake/Ceres" \
  -DCMAKE_PREFIX_PATH="$PWD/build/deps/ros_ws/devel;$PWD/build/deps/install;/opt/ros/noetic"
cmake --build build/noetic --parallel 16
```

这不是全新机器的依赖安装脚本；删除整个 `build/` 会同时删除本次构建的依赖、补丁和构建清单。

## 检查结果及记录

- SLICT 12 个目标编译、链接成功，`ldd` 未发现缺失动态库。
- 本地 Ceres 2.1 的 `SPARSE_NORMAL_CHOLESKY` 求解与 `SPARSE_QR` 协方差数值检查通过。
- ClampSLAM 的 SPLBATCH 在**相同本地 Ceres 2.1**上独立编译成功，3/3 测试通过：scaled loss、incremental、B-spline；公开头文件独立编译检查通过。
- SPLBATCH 示例的 120 个观测形成 3 个批次，优化收敛，控制点最大误差约 `3.84e-13`。这验证库的基础兼容性，不代表已经验证 SLICT 因子适配或取得 SLAM 加速结果。

重跑库验证：

```sh
ctest --test-dir build/splbatch-validation --output-on-failure
./build/ceres-smoke/build/ceres21_smoke
```

主要本地记录：

- `build/noetic-configure.log`、`build/noetic-build.log`：主项目配置与编译。
- `build/noetic-linkage.log`：所有目标的动态链接结果。
- `build/noetic-build-manifest.json`：版本、构建并行度、消息定义与二进制 SHA-256。
- `build/splbatch-validation/Testing/Temporary/LastTest.log`：SPLBATCH 测试记录。
- [效率对比方案](solver_benchmark_plan.md)：后续实现与实验设计。
