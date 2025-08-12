#include<string.h>
#include "kernel_operator.h"
using namespace AscendC;

// ==================== 算子参数配置 ====================
constexpr int32_t PILLAR_FEATURE_SIZE = 64;           // 每个pillar的特征维度
constexpr int32_t MAX_PILLARS_PER_CORE = 2048;        // 每个核心处理的最大pillar数
constexpr int32_t USE_CORE_NUM = 8;                   // 使用的核心数
constexpr int32_t BUFFER_NUM = 2;                     // 双缓冲
constexpr int32_t FEATURE_X = 1024;                    // BEV特征图宽度 (nx)
constexpr int32_t FEATURE_Y = 1024;                    // BEV特征图高度 (ny)

// 控制调试输出的开关
// constexpr bool ENABLE_DEBUG_PRINT = false;  // 关闭调试输出，提升性能

class KernelPillarScatter {
public:
    __aicore__ inline KernelPillarScatter() {}
    
    /**
     * @brief 算子初始化函数
     * 
     * 核心功能：
     * 1. 解析输入参数，获取pillar总数
     * 2. 计算当前AI Core的数据分片范围
     * 3. 设置全局内存缓冲区指针
     * 4. 初始化本地内存队列（双缓冲）
     * 5. 处理内存对齐和边界安全问题
     * 
     * @param pillar_features 输入的pillar特征数据
     *        - 数据格式: [num_pillars, PILLAR_FEATURE_SIZE] 
     *        - 数据类型: half (float16)
     *        - 物理含义: 每个pillar经过PointNet处理后的64维特征向量
     * 
     * @param coords 坐标信息数据
     *        - 数据格式: [num_pillars, 4]
     *        - 数据类型: uint32_t
     *        - coords[:, 0]: batch索引（通常为0，单batch处理）
     *        - coords[:, 1]: pillar在BEV网格中的x坐标 (0 ~ FEATURE_X-1)
     *        - coords[:, 2]: pillar在BEV网格中的y坐标 (0 ~ FEATURE_Y-1)  
     *        - coords[:, 3]: 保留字段（未使用）
     * 
     * @param params 算子参数
     *        - params[0]: 有效pillar的总数量 (uint32_t)
     *        - 用于动态确定处理规模，支持不同大小的输入
     * 
     * @param spatial_features 输出的BEV特征图
     *        - 数据格式: [1, FEATURE_Y, FEATURE_X, PILLAR_FEATURE_SIZE] (NHWC)
     *        - 数据类型: half (float16)
     *        - 物理含义: 720x720的BEV网格，每个位置存储64维特征
     *        - 初始状态: 全零，只有有pillar的位置会被填充
     */
    __aicore__ inline void Init(GM_ADDR pillar_features, GM_ADDR coords, 
                                GM_ADDR params, GM_ADDR spatial_features)
    {
        // ==================== 1. 获取当前AI Core信息 ====================
        int32_t current_block_idx = GetBlockIdx();
        
        // ==================== 2. 解析输入参数 ====================
        uint32_t total_pillars = *((__gm__ uint32_t*)params);
        this->total_pillars = total_pillars;  // 保存为成员变量，供其他函数使用
        
        // ==================== 4. 数据分片计算 ====================
        int32_t pillars_per_core = (total_pillars + USE_CORE_NUM - 1) / USE_CORE_NUM;
        
        // 计算当前Core的数据范围 [pillar_start_idx, pillar_end_idx)
        pillar_start_idx = current_block_idx * pillars_per_core;
        pillar_end_idx = Min(pillar_start_idx + pillars_per_core, total_pillars);
        num_pillars_to_process = pillar_end_idx - pillar_start_idx;
        
        // ==================== 5. 全局内存缓冲区设置 ====================
        // 5.1 设置pillar特征数据缓冲区
        // 每个Core只需要访问自己负责的pillar特征
        // 指针偏移 = pillar_start_idx * PILLAR_FEATURE_SIZE
        pillarFeaturesGm.SetGlobalBuffer((__gm__ half*)pillar_features + pillar_start_idx * PILLAR_FEATURE_SIZE, 
                                         num_pillars_to_process * PILLAR_FEATURE_SIZE);
        
        // 5.2 设置坐标数据缓冲区（简化版）
        // 每次读取4个uint32_t（1个pillar的坐标），无需复杂的对齐计算
        int32_t coords_buffer_size = num_pillars_to_process * 4 + 8 ;  // 恰好需要的坐标数量 +8防止溢出
        
        
        coordsGm.SetGlobalBuffer((__gm__ uint32_t*)coords + pillar_start_idx * 4, coords_buffer_size);
        
        // 保存缓冲区大小，供CopyIn函数使用
        // this->coords_buffer_size = coords_buffer_size;
        
        // 5.3 设置输出特征图缓冲区
        // 所有Core共享同一个输出缓冲区，但写入不同位置（无冲突）
        // NHWC格式: [1, 720, 720, 64]，同一位置的64个通道连续存储
        spatialFeaturesGm.SetGlobalBuffer((__gm__ half*)spatial_features, 1 * FEATURE_Y * FEATURE_X * PILLAR_FEATURE_SIZE);
    }
    
    __aicore__ inline void Process()
    {
        int32_t current_core_id = GetBlockIdx();
        
        for (int32_t i = 0; i < num_pillars_to_process; i++) {
            Compute(i);  // 计算处理：坐标验证 + 特征写入
        }
    }

private:
    __aicore__ inline void Compute(int32_t progress)
    {
        // ==================== 1. 获取本地数据 ====================
        int32_t current_offset = progress * 4;
        
        // ==================== 2. 解析坐标信息 ====================
        uint32_t batch = coordsGm.GetValue(current_offset + 0);  // batch索引（通常为0）
        uint32_t x = coordsGm.GetValue(current_offset + 2);      // BEV网格x坐标 [0, FEATURE_X-1] (修复：原来是GetValue(1))
        uint32_t y = coordsGm.GetValue(current_offset + 1);      // BEV网格y坐标 [0, FEATURE_Y-1] (修复：原来是GetValue(2))
        // coordsLocal.GetValue(3) 是保留字段，未使用
        
        // ==================== 6. 计算NHWC格式的输出位置 ====================
        // NHWC格式：[Batch, Height, Width, Channel]
        // 对于位置(x,y)，64个通道的特征值连续存储
        // offset公式：batch * H * W * C + y * W * C + x * C
        uint32_t base_offset = y * FEATURE_X * PILLAR_FEATURE_SIZE + x * PILLAR_FEATURE_SIZE;
        
        for (int32_t i = 0; i < PILLAR_FEATURE_SIZE; i++) {
            spatialFeaturesGm.SetValue(base_offset + i, pillarFeaturesGm.GetValue(progress * PILLAR_FEATURE_SIZE + i));
        }
    }
    
    __aicore__ inline int32_t Min(int32_t a, int32_t b)
    {
        return (a < b) ? a : b;
    }

private:
    // ==================== 流水线和队列管理 ====================
    TPipe pipe;  // 流水线管理器，协调数据传输和计算
    
    // ==================== 全局内存访问张量 ====================
    // 这些张量管理对全局内存的访问，提供了类型安全和边界检查
    GlobalTensor<half> pillarFeaturesGm;      // pillar特征数据全局内存访问器
    GlobalTensor<uint32_t> coordsGm;          // 坐标数据全局内存访问器  
    GlobalTensor<half> spatialFeaturesGm;     // 输出特征图全局内存访问器
    
    // ==================== 数据分片和处理参数 ====================
    // 当前AI Core的数据处理范围和相关信息
    int32_t pillar_start_idx;        // 当前Core处理的起始pillar索引（全局索引）
    int32_t pillar_end_idx;          // 当前Core处理的结束pillar索引（全局索引，不包含）
    int32_t num_pillars_to_process;  // 当前Core需要处理的pillar总数
    uint32_t total_pillars;          // 全局pillar总数（所有Core共享）
    
    // ==================== 内存管理和对齐参数 ====================
    // int32_t coords_buffer_size;      // 坐标缓冲区的实际大小（uint32_t个数）
};

extern "C" __global__ __aicore__ void pillar_scatter_custom(GM_ADDR pillar_features, 
                                                            GM_ADDR coords, 
                                                            GM_ADDR params, 
                                                            GM_ADDR spatial_features)
{
    KernelPillarScatter op;
    op.Init(pillar_features, coords, params, spatial_features);  // 初始化和数据分片
    op.Process();  // 执行主要的scatter操作
}

#ifndef ASCENDC_CPU_DEBUG
void pillar_scatter_do(uint32_t blockDim, void *stream, GM_ADDR pillar_features, 
                                                            GM_ADDR coords, 
                                                            GM_ADDR params, 
                                                            GM_ADDR spatial_features)
{
    pillar_scatter_custom<<<blockDim, nullptr, stream>>>(pillar_features, coords, params, spatial_features);
}
#endif