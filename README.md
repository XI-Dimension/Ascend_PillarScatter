# PillarScatterAscend

PillarScatter算子的Ascend C实现 - PointPillars 3D目标检测算法的关键组件

## 项目概述

PillarScatter算子是PointPillars 3D目标检测算法中的核心组件，用于将稀疏的pillar特征根据坐标信息scatter到稠密的BEV(Bird's Eye View)特征图中。本项目提供了高性能的Ascend C实现，具备NHWC格式优化、8核并行处理等先进特性。

### 核心功能
- **稀疏到稠密转换**: 将稀疏pillar特征[N, 64]转换为稠密BEV特征图[1, 720, 720, 64]
- **高性能并行处理**: 8个AI Core并行执行，支持高吞吐量处理
- **内存优化**: NHWC格式优化，高效的批量内存操作
- **鲁棒性**: 完善的边界检查和错误处理机制

## 目录结构介绍

```
PillarScatterKernel/
├── cmake/                       # 编译工程文件
│   ├── cpu_lib.cmake           # CPU编译配置
│   └── npu_lib.cmake           # NPU编译配置
├── scripts/                     # 辅助脚本
│   ├── gen_data.py             # 输入数据和真值数据生成脚本
│   └── verify_result.py        # 验证输出数据和真值数据是否一致的验证脚本
├── input/                       # 测试输入数据
│   ├── OpTest_scatter_input_x.bin      # pillar特征数据
│   ├── OpTest_scatter_input_coords.bin # 坐标数据
│   └── ...                     # 其他测试数据
├── output/                      # 输出结果
│   ├── OpTest_scatter_output_x.bin     # 算子输出
│   ├── golden.bin              # 参考输出
│   └── ...                     # 其他输出文件
├── visualizations/              # 可视化结果
│   └── mean_comparison.png      # 特征图对比图
├── pillar_scatter_custom.cpp    # 算子kernel实现
├── main.cpp                     # 主函数，调用算子的应用程序
├── data_utils.h                 # 数据读入写出函数
├── CMakeLists.txt              # 编译工程文件
├── run.sh                      # 编译运行算子的脚本
├── check_visual.py             # 可视化验证工具
├── fix_permissions.sh          # 权限修复脚本
└── README_visual.md            # 可视化工具说明
```

## 算子实现介绍

### 算法原理
PillarScatter算子的数学表达式为：
```
spatial_features[batch, y, x, :] = pillar_features[pillar_idx, :]
其中 (x, y) = coords[pillar_idx, 1:3]
```

### 核心技术特性

#### 1. NHWC数据格式优化
- **传统NCHW格式**: 通道维度分散，需要64次SetValue操作
- **优化NHWC格式**: 同一位置64个通道连续存储，一次DataCopy(128字节)完成
- **性能提升**: 大幅减少内存访问次数，提高Cache命中率

#### 2. 8核并行处理
- **数据分片**: 智能负载均衡，适应不同规模输入
- **并行策略**: 每个AI Core处理约1/8的pillar数据
- **无冲突写入**: 不同Core写入不同位置，无需同步

#### 3. 双缓冲机制
- **流水线处理**: 数据传输与计算并行执行
- **延迟隐藏**: 有效隐藏内存访问延迟
- **资源管理**: 高效的本地内存队列管理

### 实现流程
算子的实现流程分为3个基本任务：
1. **CopyIn**: 将Global Memory上的pillar特征和坐标数据搬运到Local Memory
2. **Compute**: 根据坐标信息将pillar特征scatter到BEV网格的对应位置
3. **自动释放**: 本地内存资源自动管理，无需显式CopyOut

具体实现请参考 [pillar_scatter_custom.cpp](./pillar_scatter_custom.cpp)

## 性能表现

- **执行时间**: ~0.8ms (处理9282个pillars)
- **吞吐量**: >11K pillars/秒
- **内存效率**: 批量操作减少内存访问次数
- **扩展性**: 支持不同规模的点云数据

## 环境要求

- Ascend C开发环境
- NPU硬件支持 (Ascend 310P/910系列)
- Python 3.x (用于可视化工具)
- OpenCV, matplotlib, numpy (Python依赖)

## 快速开始

### 1. 配置环境变量

请根据当前环境上CANN开发套件包的安装方式，选择对应配置环境变量的命令：

- **默认路径，root用户安装CANN软件包**
  ```bash
  export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
  ```

- **默认路径，非root用户安装CANN软件包**
  ```bash
  export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
  ```

- **自定义路径安装CANN软件包**
  ```bash
  export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
  ```

配置仿真模式日志文件目录：
```bash
export CAMODEL_LOG_PATH=./sim_log
```

### 2. 编译运行

```bash
# 进入项目目录
cd /path/to/PillarScatterKernel

# 使用一键运行脚本
bash run.sh -r [RUN_MODE] -v [SOC_VERSION]
```

**参数说明:**
- **RUN_MODE**: 编译方式，支持参数: 
  - `cpu` - CPU调试模式
  - `sim` - NPU仿真模式 
  - `npu` - NPU上板模式
- **SOC_VERSION**: 昇腾AI处理器型号，支持:
  - Atlas 推理系列: `Ascend310P1`、`Ascend310P3`
  - Atlas 训练系列: `AscendxxxA`、`AscendxxxB`

**示例:**
```bash
# CPU调试
bash run.sh -r cpu -v Ascend310P1

# NPU仿真
bash run.sh -r sim -v Ascend310P1

# NPU上板
bash run.sh -r npu -v Ascend310P1
```

### 3. 可视化验证

```bash
# 生成可视化结果
python check_visual.py

# 仅保存图片（远程环境推荐）
python check_visual.py --no-display

# 查看不同聚合方式
python check_visual.py --agg max    # 最大值聚合
python check_visual.py --agg sum    # 求和聚合
python check_visual.py --agg norm   # 范数聚合
```

### 4. 权限修复（如需要）

```bash
# 修复输出文件权限
bash fix_permissions.sh
```

## 工具介绍

### 可视化验证工具 (check_visual.py)
- **格式自适应**: 自动处理NHWC和NCHW格式差异
- **多种聚合方式**: mean, max, sum, norm四种特征聚合
- **交互式浏览**: 支持单通道特征图查看
- **差异分析**: Pillar位置标记和差异图显示
- **远程友好**: 支持无显示模式，自动保存图片

详细使用说明请参考: [README_visual.md](./README_visual.md)

### 性能监控
- **高精度计时**: 微秒级执行时间测量
- **详细指标**: 平均处理时间、吞吐量统计
- **进度追踪**: 实时显示处理进度

## 技术突破

### 主要优化成果
1. **内存越界修复**: 解决了coords数组32字节对齐导致的越界问题
2. **坐标轴映射修复**: 修正了x/y坐标使用错误，输出无需旋转
3. **NPU编译适配**: 解决变量名冲突和字符串类型问题
4. **性能调优**: 关闭调试输出，性能提升10-30%

### 架构特点
- **数据格式**: NHWC优化，同位置通道连续存储
- **并行设计**: 8核无冲突并行，智能负载均衡
- **内存管理**: 双缓冲流水线，自动资源清理
- **边界安全**: 完善的坐标验证和越界保护

## 测试数据

- **输入规模**: 9282个pillars，每个64维特征
- **输出规模**: 720×720×64 BEV特征图
- **数据类型**: half (float16)
- **坐标范围**: x∈[0,719], y∈[0,719]

## 开发历程

| 时间 | 更新事项 |
|------|----------|
| 2024/05/29 | 初始实现和基础功能验证 |
| 2024/05/29 | NHWC格式优化和8核并行 |
| 2024/05/30 | 内存越界问题修复 |
| 2024/05/30 | 坐标轴映射错误修复 |
| 2024/05/30 | NPU编译环境适配 |
| 2024/05/30 | 性能调优和可视化工具完善 |

## 许可证

Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 