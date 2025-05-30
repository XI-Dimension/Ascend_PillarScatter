#!/usr/bin/env python3
"""
对比PillarScatter算子输出与参考输出的脚本
用于验证算子实现的正确性
"""

import numpy as np
import os
import sys
import hashlib

# 调试模式配置
DEBUG_MODE = True  # 设置为True来启用调试模式
DEBUG_MAX_PILLARS = 9282  # 与算子代码中的限制保持一致

def load_binary_file(filename, dtype=np.float16):
    """
    加载二进制文件为numpy数组
    
    Args:
        filename: 文件路径
        dtype: 数据类型，默认为float16
    
    Returns:
        numpy数组
    """
    if not os.path.exists(filename):
        print(f"错误：文件 {filename} 不存在")
        return None
    
    try:
        data = np.fromfile(filename, dtype=dtype)
        return data
    except Exception as e:
        print(f"读取文件 {filename} 时出错：{e}")
        return None

def load_coords_file(filename="./input/OpTest_scatter_input_coords.bin"):
    """
    加载坐标文件来获取pillar的位置信息
    
    Args:
        filename: 坐标文件路径
    
    Returns:
        numpy数组，shape为[num_pillars, 4]
    """
    if not os.path.exists(filename):
        print(f"警告：坐标文件 {filename} 不存在，无法进行空间验证")
        return None
    
    try:
        coords = np.fromfile(filename, dtype=np.uint32)
        coords = coords.reshape(-1, 4)  # [num_pillars, 4] - [batch, x, y, reserved]
        return coords
    except Exception as e:
        print(f"读取坐标文件 {filename} 时出错：{e}")
        return None

def get_file_md5(filename):
    """计算文件的MD5值"""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def analyze_debug_output(output_data, coords=None):
    """
    分析调试模式下的输出数据
    
    Args:
        output_data: 输出数据 [1, 64, 720, 720]
        coords: 坐标数据 [num_pillars, 4]，可选
    
    Returns:
        dict: 分析结果
    """
    print(f"\n{'='*60}")
    print(f"调试模式分析 (限制前{DEBUG_MAX_PILLARS}个pillar)")
    print(f"{'='*60}")
    
    # 基本统计
    total_elements = output_data.size
    nonzero_elements = np.count_nonzero(output_data)
    
    print(f"总元素数: {total_elements:,}")
    print(f"非零元素数: {nonzero_elements:,} ({nonzero_elements/total_elements*100:.4f}%)")
    
    # 分析每个通道
    batch, height, width, channels = output_data.shape  # NHWC: [B, H, W, C]
    print(f"\n各通道非零统计:")
    for c in range(min(channels, 10)):  # 只显示前10个通道
        channel_data = output_data[0, :, :, c]  # NHWC格式访问第c个通道
        channel_nonzero = np.count_nonzero(channel_data)
        print(f"  通道 {c:2d}: {channel_nonzero:4d} 个非零值")
    
    if channels > 10:
        print(f"  ... (省略剩余 {channels-10} 个通道)")
    
    # 如果有坐标信息，验证写入位置
    if coords is not None:
        print(f"\n位置验证:")
        valid_coords = coords[:DEBUG_MAX_PILLARS]  # 只检查前301个
        expected_positions = 0
        actual_positions = 0
        
        for i, (batch_id, x, y, _) in enumerate(valid_coords):
            if batch_id == 0 and x < width and y < height:
                expected_positions += 1
                # 检查这个位置是否有值（检查所有通道）
                position_values = output_data[0, y, x, :]  # NHWC格式：[B, H, W, C]
                if np.any(position_values != 0):
                    actual_positions += 1
                
                # 显示前几个位置的详细信息
                if i < 5:
                    value_count = np.count_nonzero(position_values)
                    print(f"  Pillar {i}: ({x}, {y}) -> {value_count}/64 通道有值")
        
        print(f"\n预期有值位置: {expected_positions}")
        print(f"实际有值位置: {actual_positions}")
        
        if actual_positions == expected_positions:
            print("✓ 位置验证通过")
        else:
            print("✗ 位置验证失败")
    
    return {
        'total_elements': total_elements,
        'nonzero_elements': nonzero_elements,
        'nonzero_ratio': nonzero_elements / total_elements
    }

def compare_arrays(arr1, arr2, name1="输出", name2="参考", tolerance=1e-5, debug_coords=None):
    """
    比较两个numpy数组
    
    Args:
        arr1: 第一个数组
        arr2: 第二个数组
        name1: 第一个数组的名称
        name2: 第二个数组的名称
        tolerance: 容差值
        debug_coords: 调试模式下的坐标信息
    
    Returns:
        bool: 是否一致
    """
    print(f"\n{'='*60}")
    print(f"比较 {name1} 和 {name2}")
    print(f"{'='*60}")
    
    # 检查形状
    if arr1.shape != arr2.shape:
        print(f"✗ 形状不匹配！")
        print(f"  {name1}: {arr1.shape}")
        print(f"  {name2}: {arr2.shape}")
        return False
    
    print(f"✓ 形状匹配: {arr1.shape}")
    
    # 调试模式分析
    if DEBUG_MODE:
        print(f"\n{name1} 分析:")
        analyze_debug_output(arr1, debug_coords)
        
        print(f"\n{name2} 分析:")
        analyze_debug_output(arr2, debug_coords)
    
    # 统计信息
    print(f"\n{name1}统计信息:")
    print(f"  最小值: {arr1.min():.6f}")
    print(f"  最大值: {arr1.max():.6f}")
    print(f"  平均值: {arr1.mean():.6f}")
    print(f"  标准差: {arr1.std():.6f}")
    
    print(f"\n{name2}统计信息:")
    print(f"  最小值: {arr2.min():.6f}")
    print(f"  最大值: {arr2.max():.6f}")
    print(f"  平均值: {arr2.mean():.6f}")
    print(f"  标准差: {arr2.std():.6f}")
    
    # 计算差异
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\n差异分析:")
    print(f"  最大绝对差异: {max_diff:.6e}")
    print(f"  平均绝对差异: {mean_diff:.6e}")
    
    # 统计非零元素
    nonzero1 = np.count_nonzero(arr1)
    nonzero2 = np.count_nonzero(arr2)
    print(f"\n非零元素统计:")
    print(f"  {name1}: {nonzero1} ({nonzero1/arr1.size*100:.2f}%)")
    print(f"  {name2}: {nonzero2} ({nonzero2/arr2.size*100:.2f}%)")
    
    # 调试模式下的特殊比较逻辑
    if DEBUG_MODE and debug_coords is not None:
        print(f"\n调试模式比较 (仅验证前{DEBUG_MAX_PILLARS}个pillar对应的位置):")
        valid_coords = debug_coords[:DEBUG_MAX_PILLARS]
        
        match_count = 0
        total_checked = 0
        
        for i, (batch_id, x, y, _) in enumerate(valid_coords):
            if batch_id == 0 and x < arr1.shape[2] and y < arr1.shape[1]:  # NHWC: [B, H, W, C]
                total_checked += 1
                pos1 = arr1[0, y, x, :]  # NHWC格式：所有通道在该位置的值
                pos2 = arr2[0, y, x, :]
                
                if np.allclose(pos1, pos2, rtol=tolerance, atol=tolerance):
                    match_count += 1
                elif i < 5:  # 显示前几个不匹配的详细信息
                    print(f"  位置({x}, {y}) 不匹配:")
                    for c in range(min(5, len(pos1))):
                        if abs(pos1[c] - pos2[c]) > tolerance:
                            print(f"    通道{c}: {pos1[c]:.6f} vs {pos2[c]:.6f}")
        
        print(f"有效位置匹配率: {match_count}/{total_checked} ({match_count/max(total_checked,1)*100:.2f}%)")
        
        if match_count == total_checked:
            print(f"✓ 调试范围内数据一致！")
            return True
    
    # 检查完全相等
    if np.array_equal(arr1, arr2):
        print(f"\n✓ 完全一致！")
        return True
    
    # 检查近似相等
    if np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance):
        print(f"\n✓ 在容差 {tolerance} 内一致")
        return True
    
    # 找出差异较大的位置
    diff_mask = diff > tolerance
    diff_count = np.sum(diff_mask)
    
    if diff_count > 0:
        print(f"\n✗ 发现 {diff_count} 个差异超过容差的元素 ({diff_count/arr1.size*100:.2f}%)")
        
        # 显示前几个差异
        diff_indices = np.where(diff_mask)
        show_count = min(10, diff_count)
        print(f"\n显示前 {show_count} 个差异:")
        
        for i in range(show_count):
            idx = tuple(d[i] for d in diff_indices)
            print(f"  位置 {idx}: {name1}={arr1[idx]:.6f}, {name2}={arr2[idx]:.6f}, 差异={diff[idx]:.6e}")
    
    return False

def main():
    """主函数"""
    # 默认文件路径
    output_file = "./output/OpTest_scatter_output_x.bin"
    correct_file = "./output/OpTest_scatter_output_x_correct.bin"
    coords_file = "./input/OpTest_scatter_input_coords.bin"
    
    # 支持命令行参数
    if len(sys.argv) >= 3:
        output_file = sys.argv[1]
        correct_file = sys.argv[2]
    
    print("PillarScatter算子输出验证工具")
    if DEBUG_MODE:
        print(f"*** 调试模式：仅验证前{DEBUG_MAX_PILLARS}个pillar ***")
    
    print(f"\n输出文件: {output_file}")
    print(f"参考文件: {correct_file}")
    print(f"坐标文件: {coords_file}")
    
    # 检查文件是否存在
    if not os.path.exists(output_file):
        print(f"\n错误：输出文件 {output_file} 不存在")
        print("请先运行算子生成输出文件")
        return 1
    
    if not os.path.exists(correct_file):
        print(f"\n错误：参考文件 {correct_file} 不存在")
        return 1
    
    # 加载坐标文件（调试模式下需要）
    coords_data = None
    if DEBUG_MODE:
        coords_data = load_coords_file(coords_file)
        if coords_data is not None:
            print(f"✓ 加载坐标文件成功: {coords_data.shape}")
            print(f"总pillar数: {len(coords_data)}")
            if len(coords_data) > DEBUG_MAX_PILLARS:
                print(f"调试模式限制: 只处理前{DEBUG_MAX_PILLARS}个pillar")
        else:
            print("⚠ 无法加载坐标文件，将进行常规比较")
    
    # 显示文件信息
    output_size = os.path.getsize(output_file)
    correct_size = os.path.getsize(correct_file)
    
    print(f"\n文件大小:")
    print(f"  输出文件: {output_size:,} 字节")
    print(f"  参考文件: {correct_size:,} 字节")
    
    if output_size != correct_size:
        print("\n✗ 文件大小不一致！")
        return 1
    
    # 计算MD5
    print(f"\nMD5校验:")
    output_md5 = get_file_md5(output_file)
    correct_md5 = get_file_md5(correct_file)
    print(f"  输出文件: {output_md5}")
    print(f"  参考文件: {correct_md5}")
    
    if output_md5 == correct_md5:
        print("\n✓ MD5完全一致！文件内容相同")
        return 0
    
    # 调试模式提示
    if DEBUG_MODE:
        print("\n⚠ MD5不一致，但这在调试模式下是正常的")
        print("  因为我们只处理了部分数据，其他位置可能保持初始值")
    
    # 加载数据
    print("\n加载数据...")
    output_data = load_binary_file(output_file, dtype=np.float16)
    correct_data = load_binary_file(correct_file, dtype=np.float16)
    
    if output_data is None or correct_data is None:
        return 1
    
    # 预期形状 [1, 720, 720, 64] (NHWC)
    expected_shape = (1 * 720 * 720 * 64,)
    if output_data.shape[0] == expected_shape[0]:
        # 算子输出：重塑为NHWC格式
        output_data = output_data.reshape(1, 720, 720, 64)
        print(f"\n算子输出已重塑为NHWC格式: {output_data.shape}")
        
        # 参考数据：从NCHW格式转换为NHWC格式
        # 参考数据原本是NCHW: [1, 64, 720, 720]
        correct_data = correct_data.reshape(1, 64, 720, 720)  # 先重塑为NCHW
        correct_data = correct_data.transpose(0, 2, 3, 1)     # 转置为NHWC: [1, 720, 720, 64]
        print(f"参考数据已从NCHW转置为NHWC格式: {correct_data.shape}")
    else:
        print(f"警告：数据大小不匹配预期！实际大小: {output_data.shape}")
    
    # 比较数据
    is_equal = compare_arrays(output_data, correct_data, "算子输出", "参考输出", 
                             debug_coords=coords_data)
    
    # 总结
    print(f"\n{'='*60}")
    if is_equal:
        if DEBUG_MODE:
            print(f"✓ 调试验证通过！前{DEBUG_MAX_PILLARS}个pillar的NHWC输出正确")
        else:
            print("✓ 验证通过！NHWC格式算子输出正确")
        return 0
    else:
        if DEBUG_MODE:
            print(f"✗ 调试验证失败！前{DEBUG_MAX_PILLARS}个pillar的NHWC输出与参考不一致")
            print("提示：检查算子逻辑或NHWC格式转换是否正确")
        else:
            print("✗ 验证失败！NHWC格式算子输出与参考不一致")
        return 1

if __name__ == "__main__":
    exit(main()) 