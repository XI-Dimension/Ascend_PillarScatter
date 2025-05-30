#!/usr/bin/env python3
"""
PillarScatter算子可视化验证工具
============================

功能：
1. 读取NHWC格式的算子输出和NCHW格式的参考输出
2. 将720x720的特征图可视化显示
3. 支持多种可视化模式：单通道、通道平均、通道最大值等
4. 使用热力图显示特征分布，便于直观比较

作者：AI助手
日期：2024
"""

import os
import sys
import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import argparse

# 配置参数
FEATURE_SIZE = 64  # 特征通道数
FEATURE_H = 720    # 特征图高度
FEATURE_W = 720    # 特征图宽度

# 添加信号处理，支持Ctrl+C中断
def signal_handler(sig, frame):
    print('\n程序被用户中断 (Ctrl+C)')
    plt.close('all')  # 关闭所有matplotlib窗口
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class FeatureVisualizer:
    """特征图可视化器"""
    
    def __init__(self, no_display=False):
        self.coords_data = None
        self.no_display = no_display
        self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """设置matplotlib显示参数"""
        # 检测是否在远程环境或无显示环境
        import matplotlib
        if self.no_display or os.environ.get('DISPLAY') is None or 'SSH' in os.environ.get('SSH_CONNECTION', ''):
            if self.no_display:
                print("--no-display模式：只保存图片，不显示窗口")
            else:
                print("检测到远程环境，使用Agg后端（仅保存图片）")
            matplotlib.use('Agg')
        else:
            # 尝试使用交互式后端
            try:
                matplotlib.use('TkAgg')
            except:
                try:
                    matplotlib.use('Qt5Agg')
                except:
                    print("无法使用交互式后端，切换到Agg后端（仅保存图片）")
                    matplotlib.use('Agg')
        
        plt.style.use('default')
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (16, 8)
        # 设置非阻塞模式
        if not self.no_display:
            plt.ion()  # 开启交互模式
    
    def load_coords(self, coords_file):
        """加载坐标文件，用于标记pillar位置"""
        if not os.path.exists(coords_file):
            print(f"警告：坐标文件 {coords_file} 不存在")
            return None
            
        try:
            coords = np.fromfile(coords_file, dtype=np.uint32)
            coords = coords.reshape(-1, 4)  # [num_pillars, 4]
            self.coords_data = coords
            print(f"✓ 加载坐标文件: {coords.shape[0]} pillars")
            return coords
        except Exception as e:
            print(f"加载坐标文件失败: {e}")
            return None
    
    def load_feature_map(self, filename, format_type):
        """
        加载特征图文件
        
        Args:
            filename: 文件路径
            format_type: 'NHWC' 或 'NCHW'
        
        Returns:
            numpy array: [1, H, W, C] 格式的特征图
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"文件不存在: {filename}")
        
        # 读取二进制数据
        data = np.fromfile(filename, dtype=np.float16)
        total_elements = 1 * FEATURE_H * FEATURE_W * FEATURE_SIZE
        
        if len(data) != total_elements:
            print(f"警告：文件大小不匹配！期望 {total_elements} 个元素，实际 {len(data)} 个")
            if len(data) < total_elements:
                raise ValueError("文件数据不足")
            data = data[:total_elements]  # 截断多余数据
        
        # 根据格式重新组织数据
        if format_type.upper() == 'NHWC':
            # NHWC: [1, H, W, C]
            features = data.reshape(1, FEATURE_H, FEATURE_W, FEATURE_SIZE)
        elif format_type.upper() == 'NCHW':
            # NCHW: [1, C, H, W] -> [1, H, W, C]
            features = data.reshape(1, FEATURE_SIZE, FEATURE_H, FEATURE_W)
            features = features.transpose(0, 2, 3, 1)  # 转换为NHWC
        else:
            raise ValueError(f"不支持的格式: {format_type}")
        
        return features.astype(np.float32)
    
    def aggregate_channels(self, features, method='mean'):
        """
        将多通道特征聚合为单通道
        
        Args:
            features: [1, H, W, C] 格式的特征图
            method: 聚合方法 'mean', 'max', 'sum', 'norm'
        
        Returns:
            numpy array: [H, W] 格式的聚合特征图
        """
        feature_map = features[0]  # 去掉batch维度: [H, W, C]
        
        if method == 'mean':
            return np.mean(feature_map, axis=2)
        elif method == 'max':
            return np.max(feature_map, axis=2)
        elif method == 'sum':
            return np.sum(feature_map, axis=2)
        elif method == 'norm':
            return np.linalg.norm(feature_map, axis=2)
        else:
            raise ValueError(f"不支持的聚合方法: {method}")
    
    def get_single_channel(self, features, channel_idx):
        """获取指定通道的特征图"""
        if channel_idx >= FEATURE_SIZE:
            raise ValueError(f"通道索引超出范围: {channel_idx} >= {FEATURE_SIZE}")
        return features[0, :, :, channel_idx]  # [H, W]
    
    def plot_feature_comparison(self, features_1, features_2, title_1, title_2, 
                              method='mean', channel_idx=None, save_path=None):
        """
        并排显示两个特征图的比较
        
        Args:
            features_1, features_2: [1, H, W, C] 格式的特征图
            title_1, title_2: 图的标题
            method: 聚合方法或 'single' 表示单通道
            channel_idx: 单通道模式下的通道索引
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 处理特征图
        if method == 'single' and channel_idx is not None:
            map_1 = self.get_single_channel(features_1, channel_idx)
            map_2 = self.get_single_channel(features_2, channel_idx)
            method_str = f"通道 {channel_idx}"
        else:
            map_1 = self.aggregate_channels(features_1, method)
            map_2 = self.aggregate_channels(features_2, method)
            method_str = f"{method} 聚合"
        
        # 统计信息
        stats_1 = self.get_statistics(map_1)
        stats_2 = self.get_statistics(map_2)
        
        # 计算差异图
        diff_map = np.abs(map_1 - map_2)
        stats_diff = self.get_statistics(diff_map)
        
        # 统一颜色范围
        vmin = min(stats_1['min'], stats_2['min'])
        vmax = max(stats_1['max'], stats_2['max'])
        
        # 绘制第一个特征图
        im1 = axes[0].imshow(map_1, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{title_1}\n{method_str}')
        axes[0].set_xlabel(f'非零像素: {stats_1["nonzero_count"]:,} ({stats_1["nonzero_ratio"]:.2%})')
        self.add_colorbar(fig, im1, axes[0])
        
        # 绘制第二个特征图
        im2 = axes[1].imshow(map_2, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{title_2}\n{method_str}')
        axes[1].set_xlabel(f'非零像素: {stats_2["nonzero_count"]:,} ({stats_2["nonzero_ratio"]:.2%})')
        self.add_colorbar(fig, im2, axes[1])
        
        # 绘制差异图
        im3 = axes[2].imshow(diff_map, cmap='Reds')
        axes[2].set_title(f'绝对差异\n最大差异: {stats_diff["max"]:.6f}')
        axes[2].set_xlabel(f'差异像素: {np.count_nonzero(diff_map):,}')
        self.add_colorbar(fig, im3, axes[2])
        
        # 标记pillar位置（如果有坐标数据）
        if self.coords_data is not None:
            self.mark_pillar_positions(axes[:2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存: {save_path}")
        
        # 非阻塞显示
        backend = plt.get_backend()
        if backend != 'Agg' and not self.no_display:
            print("显示图像窗口，按任意键继续或关闭窗口...")
            plt.show(block=False)
            try:
                # 等待用户操作，支持Ctrl+C中断
                input("按回车键继续，或按Ctrl+C退出...")
            except KeyboardInterrupt:
                print("\n用户中断显示")
            finally:
                plt.close(fig)
        else:
            if self.no_display:
                print("no-display模式：图像已保存，跳过显示")
            else:
                print("Agg后端模式：图像已保存，无法显示窗口")
            plt.close(fig)
        
        # 打印详细统计
        self.print_comparison_stats(stats_1, stats_2, stats_diff, title_1, title_2)
    
    def add_colorbar(self, fig, im, ax):
        """为图像添加颜色条"""
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
    
    def mark_pillar_positions(self, axes, max_markers=100):
        """在特征图上标记pillar位置"""
        if self.coords_data is None:
            return
        
        # 只标记前max_markers个pillar，避免图像过于杂乱
        coords_to_mark = self.coords_data[:max_markers]
        
        for ax in axes:
            for i, (batch, x, y, _) in enumerate(coords_to_mark):
                if batch == 0 and x < FEATURE_W and y < FEATURE_H:
                    # 用小圆圈标记pillar位置
                    circle = plt.Circle((x, y), radius=2, color='red', 
                                      fill=False, linewidth=1, alpha=0.7)
                    ax.add_patch(circle)
                    
                    # 前10个pillar添加数字标签
                    if i < 10:
                        ax.text(x+3, y+3, str(i), color='red', fontsize=8, 
                               fontweight='bold', alpha=0.8)
    
    def get_statistics(self, feature_map):
        """计算特征图统计信息"""
        return {
            'min': np.min(feature_map),
            'max': np.max(feature_map), 
            'mean': np.mean(feature_map),
            'std': np.std(feature_map),
            'nonzero_count': np.count_nonzero(feature_map),
            'nonzero_ratio': np.count_nonzero(feature_map) / feature_map.size
        }
    
    def print_comparison_stats(self, stats_1, stats_2, stats_diff, title_1, title_2):
        """打印比较统计信息"""
        print(f"\n{'='*60}")
        print(f"特征图统计比较")
        print(f"{'='*60}")
        
        print(f"\n{title_1}:")
        print(f"  范围: [{stats_1['min']:.6f}, {stats_1['max']:.6f}]")
        print(f"  均值±标准差: {stats_1['mean']:.6f} ± {stats_1['std']:.6f}")
        print(f"  非零像素: {stats_1['nonzero_count']:,} ({stats_1['nonzero_ratio']:.2%})")
        
        print(f"\n{title_2}:")
        print(f"  范围: [{stats_2['min']:.6f}, {stats_2['max']:.6f}]")
        print(f"  均值±标准差: {stats_2['mean']:.6f} ± {stats_2['std']:.6f}")
        print(f"  非零像素: {stats_2['nonzero_count']:,} ({stats_2['nonzero_ratio']:.2%})")
        
        print(f"\n差异统计:")
        print(f"  最大绝对差异: {stats_diff['max']:.6f}")
        print(f"  平均绝对差异: {stats_diff['mean']:.6f}")
        print(f"  差异标准差: {stats_diff['std']:.6f}")
        print(f"  有差异的像素: {np.count_nonzero(stats_diff['max'] > 0):,}")
    
    def interactive_channel_explorer(self, features_1, features_2, title_1, title_2):
        """交互式通道浏览器"""
        print(f"\n{'='*60}")
        print(f"交互式通道浏览器")
        print(f"{'='*60}")
        print(f"输入通道索引 (0-{FEATURE_SIZE-1}) 或聚合方法:")
        print(f"  0-{FEATURE_SIZE-1}: 查看指定通道")
        print(f"  mean: 通道平均值")
        print(f"  max: 通道最大值") 
        print(f"  norm: 通道范数")
        print(f"  sum: 通道求和")
        print(f"  quit/q/exit: 退出")
        print(f"  Ctrl+C: 强制退出")
        
        while True:
            try:
                user_input = input(f"\n请选择 (默认: mean): ").strip()
                
                if user_input.lower() in ['quit', 'q', 'exit', '']:
                    if user_input == '':
                        user_input = 'mean'
                        # 只显示一次默认结果就退出
                        self.plot_feature_comparison(
                            features_1, features_2, title_1, title_2,
                            method=user_input.lower(),
                            save_path=f'{user_input}_comparison.png'
                        )
                    break
                
                if user_input.isdigit():
                    channel_idx = int(user_input)
                    if 0 <= channel_idx < FEATURE_SIZE:
                        self.plot_feature_comparison(
                            features_1, features_2, title_1, title_2,
                            method='single', channel_idx=channel_idx,
                            save_path=f'channel_{channel_idx}_comparison.png'
                        )
                    else:
                        print(f"通道索引超出范围: {channel_idx}")
                elif user_input.lower() in ['mean', 'max', 'norm', 'sum']:
                    self.plot_feature_comparison(
                        features_1, features_2, title_1, title_2,
                        method=user_input.lower(),
                        save_path=f'{user_input}_comparison.png'
                    )
                else:
                    print(f"无效输入: {user_input}")
                    
            except KeyboardInterrupt:
                print("\n用户中断，退出浏览器")
                break
            except EOFError:
                print("\n检测到EOF，退出浏览器")
                break
            except Exception as e:
                print(f"错误: {e}")
        
        print("交互式浏览器已关闭")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PillarScatter特征图可视化工具')
    parser.add_argument('--output', default='./output/OpTest_scatter_output_x.bin',
                       help='算子输出文件 (NHWC格式)')
    parser.add_argument('--reference', default='./output/OpTest_scatter_output_x_correct.bin',
                       help='参考输出文件 (NCHW格式)')
    parser.add_argument('--coords', default='./input/OpTest_scatter_input_coords.bin',
                       help='坐标文件')
    parser.add_argument('--method', default='mean', 
                       choices=['mean', 'max', 'sum', 'norm'],
                       help='特征聚合方法')
    parser.add_argument('--channel', type=int, 
                       help='显示指定通道 (0-63)')
    parser.add_argument('--interactive', action='store_true',
                       help='启动交互式模式')
    parser.add_argument('--save-dir', default='./visualizations',
                       help='图像保存目录')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示图像窗口，只保存图片')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("PillarScatter特征图可视化工具")
    print("="*50)
    
    # 创建可视化器
    visualizer = FeatureVisualizer(args.no_display)
    
    # 加载坐标文件
    visualizer.load_coords(args.coords)
    
    try:
        # 加载特征图
        print(f"\n加载算子输出 (NHWC): {args.output}")
        features_output = visualizer.load_feature_map(args.output, 'NHWC')
        
        print(f"加载参考输出 (NCHW): {args.reference}")
        features_reference = visualizer.load_feature_map(args.reference, 'NCHW')
        
        print(f"✓ 成功加载特征图: {features_output.shape}")
        
        if args.interactive:
            # 交互式模式
            visualizer.interactive_channel_explorer(
                features_output, features_reference,
                "算子输出 (NHWC)", "参考输出 (NCHW)"
            )
        else:
            # 单次比较模式
            if args.channel is not None:
                save_path = os.path.join(args.save_dir, f'channel_{args.channel}_comparison.png')
                visualizer.plot_feature_comparison(
                    features_output, features_reference,
                    "算子输出 (NHWC)", "参考输出 (NCHW)",
                    method='single', channel_idx=args.channel,
                    save_path=save_path
                )
            else:
                save_path = os.path.join(args.save_dir, f'{args.method}_comparison.png')
                visualizer.plot_feature_comparison(
                    features_output, features_reference,
                    "算子输出 (NHWC)", "参考输出 (NCHW)",
                    method=args.method,
                    save_path=save_path
                )
        
    except FileNotFoundError as e:
        print(f"\n错误：{e}")
        print("请确保已运行算子生成输出文件")
        return 1
    except KeyboardInterrupt:
        print(f"\n程序被用户中断")
        return 1
    except Exception as e:
        print(f"\n处理过程中出错：{e}")
        return 1
    finally:
        # 确保所有matplotlib资源被释放
        plt.close('all')
        plt.ioff()  # 关闭交互模式
    
    print(f"\n✓ 可视化完成！图像保存在: {args.save_dir}")
    return 0

if __name__ == '__main__':
    exit(main()) 