#!/bin/bash
# 修复PillarScatter输出文件权限的脚本

echo "修复PillarScatter输出文件权限..."

# 检查文件是否存在
if [ ! -f "output/OpTest_scatter_output_x.bin" ]; then
    echo "错误：输出文件 output/OpTest_scatter_output_x.bin 不存在"
    echo "请先运行算子生成输出文件：sudo -E bash run.sh -r npu -v Ascend310P1"
    exit 1
fi

# 显示当前权限
echo "修复前的文件权限："
ls -la output/OpTest_scatter_output_x.bin

# 修复权限
echo "正在修复权限..."
sudo chmod 644 output/OpTest_scatter_output_x.bin

# 如果参考文件存在，也修复其权限
if [ -f "output/OpTest_scatter_output_x_correct.bin" ]; then
    sudo chmod 644 output/OpTest_scatter_output_x_correct.bin
fi

# 显示修复后的权限
echo "修复后的文件权限："
ls -la output/OpTest_scatter_output_x*.bin

echo "✓ 权限修复完成！现在可以运行可视化工具了："
echo "  python check_visual.py --no-display"
echo "  python check_visual.py --channel 0 --no-display"
echo "  python check_visual.py --interactive" 