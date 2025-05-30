# PillarScatter特征图可视化工具使用说明

## 功能介绍

这个可视化工具可以将NHWC和NCHW格式的720×720特征图以热力图形式显示，支持：

- 自动处理不同数据格式（NHWC vs NCHW）
- 多种特征聚合方式（平均值、最大值、范数等）
- 单通道特征图查看
- Pillar位置标记
- 差异图显示
- 交互式通道浏览

## 基本使用

### 1. 安装依赖
```bash
pip install numpy matplotlib
```

### 2. 基本命令
```bash
# 显示平均特征图
python check_visual.py

# 显示特定通道
python check_visual.py --channel 0

# 使用最大值聚合
python check_visual.py --method max

# 交互式模式
python check_visual.py --interactive
```

### 3. 参数说明
- `--output`: 算子输出文件（NHWC格式，默认：`./output/OpTest_scatter_output_x.bin`）
- `--reference`: 参考输出文件（NCHW格式，默认：`./output/OpTest_scatter_output_x_correct.bin`）
- `--coords`: 坐标文件（默认：`./input/OpTest_scatter_input_coords.bin`）
- `--method`: 聚合方法 `[mean|max|sum|norm]`（默认：`mean`）
- `--channel`: 指定通道索引 `0-63`
- `--interactive`: 启动交互式模式
- `--save-dir`: 图像保存目录（默认：`./visualizations`）
- `--no-display`: 不显示图像窗口，只保存图片（推荐用于远程服务器）

## 使用示例

### 快速查看平均特征图
```bash
python check_visual.py
```
这会显示三个图：
- 左：算子输出的平均特征图（NHWC）
- 中：参考输出的平均特征图（NCHW）  
- 右：两者的绝对差异图

### 远程服务器使用（推荐）
```bash
# 只保存图片，不显示窗口
python check_visual.py --no-display

# 查看特定通道并只保存图片
python check_visual.py --channel 0 --no-display
```

### 查看特定通道
```bash
# 查看第0通道
python check_visual.py --channel 0

# 查看第32通道
python check_visual.py --channel 32
```

### 不同聚合方法
```bash
# 最大值聚合（突出显示最强特征）
python check_visual.py --method max

# 范数聚合（显示特征强度）
python check_visual.py --method norm

# 求和聚合
python check_visual.py --method sum
```

### 交互式浏览
```bash
python check_visual.py --interactive
```
进入交互模式后，可以输入：
- `0-63`: 查看指定通道
- `mean`: 平均值聚合
- `max`: 最大值聚合
- `norm`: 范数聚合
- `sum`: 求和聚合
- `quit`: 退出

## 输出说明

### 可视化图像
- **颜色映射**：使用viridis颜色图，深蓝色表示低值，黄色表示高值
- **红圈标记**：前100个pillar的位置标记
- **数字标签**：前10个pillar的编号
- **颜色条**：显示数值范围

### 统计信息
程序会输出详细的统计信息：
- 数值范围 [min, max]
- 均值±标准差
- 非零像素数量和比例
- 最大绝对差异
- 差异统计信息

### 保存的文件
所有图像自动保存到 `./visualizations/` 目录：
- `mean_comparison.png`: 平均值比较图
- `channel_X_comparison.png`: 第X通道比较图
- `max_comparison.png`: 最大值比较图
- 等等

## 预期效果

正常情况下，你应该看到：
1. **算子输出和参考输出的特征图形状相似**
2. **有特征值的位置对应pillar的坐标**
3. **差异图大部分区域为深色（差异小）**
4. **非零像素数量与pillar数量相关**

如果看到：
- 大量红色差异区域 → 算法可能有错误
- 完全不同的特征分布 → 数据格式或算法问题
- 全零输出 → 算子未正常执行

## 故障排除

### 文件不存在错误
确保已运行算子生成输出文件：
```bash
./run.sh  # 或者你的编译运行脚本
```

### 程序无法中断（Ctrl+C不响应）
如果遇到Ctrl+C无法中断程序的问题：
```bash
# 方案1：使用--no-display模式（推荐）
python check_visual.py --no-display

# 方案2：直接关闭终端窗口
# 方案3：在另一个终端用kill命令强制结束
```

现在的版本已经修复了Ctrl+C响应问题，支持：
- 信号处理：正确响应Ctrl+C中断
- 非阻塞显示：图像窗口不会阻塞程序
- 优雅退出：自动清理matplotlib资源

### 显示问题
如果无法显示图像（远程服务器等）：
```bash
# 使用--no-display参数
python check_visual.py --no-display
```
- 图像会自动保存到文件
- 可以下载png文件查看
- 不依赖显示环境

### 内存错误
如果图像太大导致内存不足，可以：
- 减少pillar标记数量
- 使用单通道查看模式
- 使用--no-display模式减少内存占用

## 高级用法

### 批量生成所有通道
```bash
for i in {0..63}; do
    python check_visual.py --channel $i --save-dir ./all_channels
done
```

### 自定义文件路径
```bash
python check_visual.py \
    --output ./my_output.bin \
    --reference ./my_reference.bin \
    --coords ./my_coords.bin \
    --save-dir ./my_results
```

这个工具让你能够直观地看到720×720特征图的内容，通过颜色变化快速识别特征分布和算子正确性！ 