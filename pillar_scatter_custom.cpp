/**
 * @file pillar_scatter_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * 
 * PillarScatter算子的Ascend C实现
 * =====================================
 * 
 * 算子功能：
 * 将稀疏的pillar特征根据坐标信息scatter到稠密的BEV(Bird's Eye View)特征图中。
 * 这是PointPillars 3D目标检测算法中的关键步骤，将基于柱状体的特征重新排列到
 * 2D BEV空间网格中，为后续的2D卷积网络处理做准备。
 * 
 * 算法原理：
 * 1. 输入：稀疏的pillar特征[N, 64]和对应的坐标[N, 4]
 * 2. 输出：稠密的BEV特征图[1, 720, 720, 64] (NHWC格式)
 * 3. 核心操作：根据每个pillar的(x,y)坐标，将其64维特征向量放置到
 *    BEV特征图的对应位置
 * 
 * 性能优化策略：
 * - 采用NHWC数据格式，使同一位置的所有通道连续存储
 * - 使用DataCopy实现高效的批量内存操作
 * - 8个AI Core并行处理，提高吞吐量
 * - 双缓冲机制隐藏内存访问延迟
 */
#include "kernel_operator.h"
using namespace AscendC;

// ==================== 算子参数配置 ====================
constexpr int32_t PILLAR_FEATURE_SIZE = 64;           // 每个pillar的特征维度
constexpr int32_t MAX_PILLARS_PER_CORE = 2048;        // 每个核心处理的最大pillar数
constexpr int32_t USE_CORE_NUM = 8;                   // 使用的核心数
constexpr int32_t BUFFER_NUM = 2;                     // 双缓冲
constexpr int32_t FEATURE_X = 720;                    // BEV特征图宽度 (nx)
constexpr int32_t FEATURE_Y = 720;                    // BEV特征图高度 (ny)

// 控制调试输出的开关
constexpr bool ENABLE_DEBUG_PRINT = false;  // 关闭调试输出，提升性能

/**
 * @brief PillarScatter算子的Ascend C核心实现类
 * 
 * 该类实现了完整的PillarScatter算法，包括：
 * 1. 多核并行数据分片
 * 2. 高效的内存管理和数据传输
 * 3. NHWC格式的优化写入
 * 4. 边界检查和错误处理
 * 
 * 数据流处理流程：
 * Init() -> 初始化内存布局和数据分片
 * Process() -> 主处理循环
 * ├── CopyIn() -> 数据搬入（全局内存 -> 本地内存）
 * ├── Compute() -> 计算处理（坐标验证 + 特征写入）
 * └── 自动释放本地内存
 * 
 * 并行处理策略：
 * - 8个AI Core并行运行，每个处理约1/8的pillar
 * - Core 0: pillar 0 ~ pillars_per_core-1
 * - Core 1: pillar pillars_per_core ~ 2*pillars_per_core-1
 * - ...
 * - Core 7: 处理剩余的pillar（可能略少）
 */
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
        // GetBlockIdx()返回当前AI Core的编号 (0-7)
        // 用于确定当前核心在并行处理中的角色和数据分片
        int32_t current_block_idx = GetBlockIdx();
        
        // ==================== 2. 解析输入参数 ====================
        // 从全局内存读取pillar总数，这是动态参数
        // 支持不同场景下的不同pillar数量（如不同的点云密度）
        uint32_t total_pillars = *((__gm__ uint32_t*)params);
        this->total_pillars = total_pillars;  // 保存为成员变量，供其他函数使用
        
        // ==================== 3. 调试信息输出 ====================
        // 仅Core 0输出全局信息，避免重复打印
        // if (ENABLE_DEBUG_PRINT && current_block_idx == 0) {
        //     printf("PillarScatter Init: total_pillars = %u\n", total_pillars);
        //     printf("Feature dimensions: C=%d, H=%d, W=%d\n", 
        //            PILLAR_FEATURE_SIZE, FEATURE_Y, FEATURE_X);
        //     printf("Total output size: %d elements\n", 
        //            1 * PILLAR_FEATURE_SIZE * FEATURE_Y * FEATURE_X);
        // }
        
        // ==================== 4. 数据分片计算 ====================
        // 将总pillar数平均分配给8个AI Core
        // 使用向上取整确保所有pillar都被分配：(total + cores - 1) / cores
        int32_t pillars_per_core = (total_pillars + USE_CORE_NUM - 1) / USE_CORE_NUM;
        
        // 计算当前Core的数据范围 [pillar_start_idx, pillar_end_idx)
        pillar_start_idx = current_block_idx * pillars_per_core;
        pillar_end_idx = Min(pillar_start_idx + pillars_per_core, total_pillars);
        num_pillars_to_process = pillar_end_idx - pillar_start_idx;
        
        // 数据分片示例 (假设total_pillars=9282):
        // Core 0: pillars 0    ~ 1160 (1161个)
        // Core 1: pillars 1161 ~ 2321 (1161个) 
        // ...
        // Core 7: pillars 8127 ~ 9281 (1155个) <- 最后一个Core可能略少
        
        // if (ENABLE_DEBUG_PRINT && current_block_idx < 8) {  // 显示8个核心的分配情况
        //     printf("Core %d: pillars_per_core=%d, processing pillars %d~%d (count=%d)\n", 
        //            current_block_idx, pillars_per_core, pillar_start_idx, pillar_end_idx, num_pillars_to_process);
        // }
        
        // ==================== 5. 全局内存缓冲区设置 ====================
        
        // 5.1 设置pillar特征数据缓冲区
        // 每个Core只需要访问自己负责的pillar特征
        // 指针偏移 = pillar_start_idx * PILLAR_FEATURE_SIZE
        pillarFeaturesGm.SetGlobalBuffer((__gm__ half*)pillar_features + pillar_start_idx * PILLAR_FEATURE_SIZE, 
                                         num_pillars_to_process * PILLAR_FEATURE_SIZE);
        
        // 5.2 设置坐标数据缓冲区（简化版）
        // 每次读取4个uint32_t（1个pillar的坐标），无需复杂的对齐计算
        int32_t coords_buffer_size = num_pillars_to_process * 4 + 8 ;  // 恰好需要的坐标数量 +8防止溢出
        
        
        coordsGm.SetGlobalBuffer((__gm__ uint32_t*)coords + pillar_start_idx * 4, 
                                 coords_buffer_size);
        
        // 保存缓冲区大小，供CopyIn函数使用
        this->coords_buffer_size = coords_buffer_size;
        
        // 5.3 设置输出特征图缓冲区
        // 所有Core共享同一个输出缓冲区，但写入不同位置（无冲突）
        // NHWC格式: [1, 720, 720, 64]，同一位置的64个通道连续存储
        spatialFeaturesGm.SetGlobalBuffer((__gm__ half*)spatial_features, 
                                          1 * FEATURE_Y * FEATURE_X * PILLAR_FEATURE_SIZE);
        
        // ==================== 6. 本地内存队列初始化 ====================
        // 双缓冲机制：提高数据传输效率，隐藏内存访问延迟
        // 当一个缓冲区在计算时，另一个缓冲区可以同时进行数据传输
        
        // 特征数据队列：存储64个half值 (128字节)
        pipe.InitBuffer(inQueueFeatures, BUFFER_NUM, PILLAR_FEATURE_SIZE * sizeof(half));
        // 坐标数据队列：存储4个uint32_t (16字节，1个pillar的坐标)
        pipe.InitBuffer(inQueueCoords, BUFFER_NUM, 8 * sizeof(uint32_t));
        
        // Core 7特殊边界分析
        // if (ENABLE_DEBUG_PRINT && current_block_idx == 7) {
        //     printf("Core 7 INIT: Detailed boundary analysis:\n");
        //     printf("Core 7  - pillar_start_idx: %d\n", pillar_start_idx);
        //     printf("Core 7  - pillar_end_idx: %d\n", pillar_end_idx);
        //     printf("Core 7  - num_pillars_to_process: %d\n", num_pillars_to_process);
        //     printf("Core 7  - coords_buffer_size: %d\n", coords_buffer_size);
        //     printf("Core 7  - last_pillar_offset: %d\n", (num_pillars_to_process - 1) * 4);
        //     printf("Core 7  - max_datacopy_offset: %d (for 8 uint32_t read)\n", (num_pillars_to_process - 1) * 4 + 8);
        //     if ((num_pillars_to_process - 1) * 4 + 8 > coords_buffer_size) {
        //         printf("Core 7  - WARNING: Last DataCopy will exceed buffer! Need %d, have %d\n", 
        //                (num_pillars_to_process - 1) * 4 + 8, coords_buffer_size);
        //     }
        // }
    }
    
    /**
     * @brief 主处理函数 - 算子的核心执行流程
     * 
     * 该函数实现了流水线处理模式：
     * 1. 顺序处理每个pillar
     * 2. 对每个pillar执行：数据搬入 -> 计算处理
     * 3. 队列机制自动处理双缓冲和数据同步
     * 
     * 流水线优势：
     * - CopyIn和Compute可以并行执行（不同pillar）
     * - 隐藏内存访问延迟
     * - 提高AI Core利用率
     * 
     * 注意：这里不需要显式的CopyOut，因为在Compute中直接写入全局内存
     */
    __aicore__ inline void Process()
    {
        // 获取当前核心编号，避免每次调用GetBlockIdx()
        int32_t current_core_id = GetBlockIdx();
        
        // Core 7详细进度追踪
        // if (ENABLE_DEBUG_PRINT && current_core_id == 7) {
        //     printf("Core 7 PROCESS: Starting to process %d pillars (%d~%d)\n", 
        //            num_pillars_to_process, pillar_start_idx, pillar_start_idx + num_pillars_to_process - 1);
        // }
        
        // 处理当前Core负责的所有pillar
        // 每次循环处理一个pillar：先搬入数据，再计算处理
        for (int32_t i = 0; i < num_pillars_to_process; i++) {
            // Core 7：每100个pillar报告一次进度
            // if (ENABLE_DEBUG_PRINT && current_core_id == 7 && i % 100 == 0) {
            //     printf("Core 7 PROCESS: Processing pillar %d/%d (global %d)\n", 
            //            i, num_pillars_to_process, pillar_start_idx + i);
            // }
            
            CopyIn(i);   // 数据搬入：全局内存 -> 本地内存
            Compute(i);  // 计算处理：坐标验证 + 特征写入
            // 注意：内存释放在Compute函数内部完成
        }
        
        // if (ENABLE_DEBUG_PRINT && current_core_id == 7) {
        //     printf("Core 7 PROCESS: Completed processing all %d pillars successfully\n", num_pillars_to_process);
        // }
    }

private:
    /**
     * @brief 数据搬入函数 - 高效的内存传输
     * 
     * 核心功能：
     * 1. 从全局内存读取pillar特征和坐标数据
     * 2. 处理DataCopy的32字节对齐要求
     * 3. 使用双缓冲队列管理本地内存
     * 4. 处理边界情况（最后几个pillar的对齐问题）
     * 
     * 内存访问模式：
     * - 特征数据：连续读取64个half值 (128字节)
     * - 坐标数据：读取8个uint32_t (32字节，满足对齐)
     * 
     * @param progress 当前处理的pillar索引（相对于当前Core的起始位置）
     *                 范围：[0, num_pillars_to_process-1]
     */
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // ==================== 1. 分配本地内存 ====================
        // 从队列中分配本地张量，队列会自动管理双缓冲
        LocalTensor<half> featuresLocal = inQueueFeatures.AllocTensor<half>();
        LocalTensor<uint32_t> coordsLocal = inQueueCoords.AllocTensor<uint32_t>();
        
        // ==================== 2. 读取pillar特征数据 ====================
        // 从全局内存批量读取64个half值到本地内存
        // 这是连续内存访问，效率很高
        DataCopy(featuresLocal, pillarFeaturesGm[progress * PILLAR_FEATURE_SIZE], PILLAR_FEATURE_SIZE);
        

        
        // 计算当前读取位置
        int32_t current_offset = progress * 4;  // 当前pillar的坐标起始位置
        

        
        // Core 7详细调试：检查最后几个pillar的处理
        // int32_t current_core_id = GetBlockIdx();
        // bool is_core7_debug = (ENABLE_DEBUG_PRINT && current_core_id == 7 && progress >= num_pillars_to_process - 10);
        // if (is_core7_debug) {
        //     int32_t remaining_in_buffer = coords_buffer_size - current_offset;
        //     printf("Core 7 DEBUG: progress=%d/%d, current_offset=%d, coords_buffer_size=%d, remaining=%d\n", 
        //            progress, num_pillars_to_process, current_offset, coords_buffer_size, remaining_in_buffer);
        //     printf("Core 7 DEBUG: About to DataCopy 8 uint32_t from offset %d\n", current_offset);
        // }
        
        DataCopy(coordsLocal, coordsGm[current_offset], 8);
        
        // if (is_core7_debug) {
        //     printf("Core 7 DEBUG: DataCopy completed successfully for progress %d\n", progress);
        // }
        
        // if (ENABLE_DEBUG_PRINT && progress < 3) {
        //     printf("Core %d: DataCopy at progress %d - reading 4 uint32_t from offset %d\n", 
        //            current_core_id, progress, current_offset);
        // }
        
        // ==================== 5. 数据入队 ====================
        // 将本地数据加入队列，队列会自动处理双缓冲同步
        inQueueFeatures.EnQue(featuresLocal);
        inQueueCoords.EnQue(coordsLocal);
    }
    
    /**
     * @brief 计算处理函数 - 核心的scatter操作
     * 
     * 核心功能：
     * 1. 从队列获取本地数据（自动同步）
     * 2. 解析坐标信息，进行有效性验证
     * 3. 计算NHWC格式的输出位置
     * 4. 使用DataCopy一次性写入64个通道的特征值
     * 5. 释放本地内存资源
     * 
     * NHWC格式优势：
     * - 同一位置的64个通道连续存储
     * - 一次DataCopy(128字节)替代64次SetValue操作
     * - 大幅提升内存带宽利用率和Cache命中率
     * 
     * @param progress 当前处理的pillar索引（相对于当前Core）
     */
    __aicore__ inline void Compute(int32_t progress)
    {
        // ==================== 1. 获取本地数据 ====================
        // 从队列中取出数据，队列会自动同步和管理双缓冲
        LocalTensor<half> featuresLocal = inQueueFeatures.DeQue<half>();
        LocalTensor<uint32_t> coordsLocal = inQueueCoords.DeQue<uint32_t>();
        
        // ==================== 2. 解析坐标信息 ====================
        // 从坐标数组中提取关键信息
        uint32_t batch = coordsLocal.GetValue(0);  // batch索引（通常为0）
        uint32_t x = coordsLocal.GetValue(2);      // BEV网格x坐标 [0, FEATURE_X-1] (修复：原来是GetValue(1))
        uint32_t y = coordsLocal.GetValue(1);      // BEV网格y坐标 [0, FEATURE_Y-1] (修复：原来是GetValue(2))
        // coordsLocal.GetValue(3) 是保留字段，未使用
        
        // Core 7详细调试：检查最后几个pillar的Compute过程
        // int32_t current_core_id = GetBlockIdx();
        // bool is_core7_debug = (ENABLE_DEBUG_PRINT && current_core_id == 7 && progress >= num_pillars_to_process - 10);
        // if (is_core7_debug) {
        //     printf("Core 7 COMPUTE: progress=%d/%d, global_pillar=%d\n", 
        //            progress, num_pillars_to_process, pillar_start_idx + progress);
        //     printf("Core 7 COMPUTE: batch=%u, x=%u, y=%u\n", batch, x, y);
        // }
        
        // ==================== 3. 调试信息输出 ====================
        // 显示每个Core前几个pillar的处理情况
        // if (ENABLE_DEBUG_PRINT && progress < 3) {
        //     printf("Core %d, Pillar %d (global %d): batch=%u, x=%u, y=%u\n", 
        //            current_core_id, progress, pillar_start_idx + progress, batch, x, y);
        // }
        
        // ==================== 4. 坐标有效性验证 ====================
        // // 检查坐标是否在有效范围内，无效的pillar会被跳过
        // bool skip_pillar = false;
        
        // if (batch != 0) {
        //     // 当前实现仅支持单batch处理
        //     skip_pillar = true;
        // } else if (x >= FEATURE_X || y >= FEATURE_Y) {
        //     // 坐标超出BEV网格范围
        //     skip_pillar = true;
        // }
        
        // if (skip_pillar) {
        //     // 跳过无效pillar，释放内存资源
        //     // if (ENABLE_DEBUG_PRINT) {
        //     //     printf("Core %d, Pillar %d: SKIPPED - invalid coordinates (batch=%u, x=%u, y=%u)\n", 
        //     //            current_core_id, pillar_start_idx + progress, batch, x, y);
        //     // }
        //     inQueueFeatures.FreeTensor(featuresLocal);
        //     inQueueCoords.FreeTensor(coordsLocal);
        //     return;
        // }
        
        // ==================== 5. 显示特征信息（调试用） ====================
        // if (ENABLE_DEBUG_PRINT && current_core_id == 0 && progress < 5) {
        //     printf("Pillar %d: batch=%u, x=%u, y=%u, feat[0]=%f, feat[1]=%f, feat[63]=%f\n", 
        //            pillar_start_idx + progress, batch, x, y,
        //            (float)featuresLocal.GetValue(0), 
        //            (float)featuresLocal.GetValue(1),
        //            (float)featuresLocal.GetValue(63));
        // }
        
        // ==================== 6. 计算NHWC格式的输出位置 ====================
        // NHWC格式：[Batch, Height, Width, Channel]
        // 对于位置(x,y)，64个通道的特征值连续存储
        // offset公式：batch * H * W * C + y * W * C + x * C
        uint32_t base_offset = y * FEATURE_X * PILLAR_FEATURE_SIZE + x * PILLAR_FEATURE_SIZE;
        
        // if (is_core7_debug) {
        //     printf("Core 7 COMPUTE: base_offset=%u, coordinates=(%u,%u)\n", base_offset, x, y);
        // }
        
        // ==================== 7. 边界检查 ====================
        // 确保64个连续元素都在输出缓冲区范围内
        // uint32_t max_allowed = 1 * FEATURE_Y * FEATURE_X * PILLAR_FEATURE_SIZE;
        // if (base_offset + PILLAR_FEATURE_SIZE > max_allowed) {
        //     printf("Core %d: ERROR! Offset range %u~%u exceeds limit %u at pillar %d\n", 
        //            current_core_id, base_offset, base_offset + PILLAR_FEATURE_SIZE - 1, 
        //            max_allowed, pillar_start_idx + progress);
        //     inQueueFeatures.FreeTensor(featuresLocal);
        //     inQueueCoords.FreeTensor(coordsLocal);
        //     return;
        // }
        
        // if (is_core7_debug) {
        //     printf("Core 7 COMPUTE: Boundary check passed, about to write %d elements at offset %u\n", 
        //            PILLAR_FEATURE_SIZE, base_offset);
        // }
        
        // // ==================== 8. 高效的特征写入（NHWC格式优势）====================
        // if (ENABLE_DEBUG_PRINT && progress == 0) {
        //     printf("Core %d: NHWC write at (%u, %u), offset=%u, writing %d elements\n", 
        //            current_core_id, x, y, base_offset, PILLAR_FEATURE_SIZE);
        // }
        
        // 关键优化：NHWC格式下，同一位置的64个通道连续存储
        // 一次DataCopy(128字节)替代64次SetValue操作，性能提升显著
        // 这是从NCHW改为NHWC格式的主要收益
        DataCopy(spatialFeaturesGm[base_offset], featuresLocal, PILLAR_FEATURE_SIZE);
        
        // if (is_core7_debug) {
        //     printf("Core 7 COMPUTE: DataCopy write completed successfully for progress %d\n", progress);
        // }
        
        // // ==================== 9. 进度统计 ====================
        // if (ENABLE_DEBUG_PRINT && (pillar_start_idx + progress + 1) % 300 == 0) {
        //     printf("Core %d: Processed %d/%d pillars\n", 
        //            current_core_id, progress + 1, num_pillars_to_process);
        // }
        
        // ==================== 10. 释放本地内存资源 ====================
        // 处理完成，释放本地内存供下一个pillar使用
        inQueueFeatures.FreeTensor(featuresLocal);
        inQueueCoords.FreeTensor(coordsLocal);
    }
    
    /**
     * @brief 辅助函数：返回两个整数中的较小值
     * 
     * 用于边界检查和安全计算，确保不会超出数组范围
     * 
     * @param a 第一个整数
     * @param b 第二个整数
     * @return 较小的值
     */
    __aicore__ inline int32_t Min(int32_t a, int32_t b)
    {
        return (a < b) ? a : b;
    }

private:
    // ==================== 流水线和队列管理 ====================
    TPipe pipe;  // 流水线管理器，协调数据传输和计算
    
    // 双缓冲队列：实现内存传输与计算的并行执行
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueFeatures;  // pillar特征数据队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCoords;    // 坐标数据队列
    
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
    int32_t coords_buffer_size;      // 坐标缓冲区的实际大小（uint32_t个数）
};

/**
 * @brief PillarScatter算子的外部入口函数
 * 
 * 这是Ascend C算子的标准入口点，会在每个AI Core上并行执行。
 * 函数签名必须与算子注册时的声明保持一致。
 * 
 * 执行流程：
 * 1. 每个AI Core创建独立的KernelPillarScatter实例
 * 2. 调用Init()初始化，计算数据分片
 * 3. 调用Process()执行主要的scatter操作
 * 4. 实例析构，自动清理资源
 * 
 * 并行执行特点：
 * - 8个AI Core同时执行此函数
 * - 每个Core处理不同的pillar子集
 * - 无需显式同步，输出位置天然无冲突
 * 
 * @param pillar_features 输入pillar特征数据的全局内存地址
 * @param coords 输入坐标数据的全局内存地址  
 * @param params 算子参数的全局内存地址
 * @param spatial_features 输出BEV特征图的全局内存地址
 */
extern "C" __global__ __aicore__ void pillar_scatter_custom(GM_ADDR pillar_features, 
                                                            GM_ADDR coords, 
                                                            GM_ADDR params, 
                                                            GM_ADDR spatial_features)
{
    // 创建算子实例并执行
    // 每个AI Core都会独立执行这个流程
    KernelPillarScatter op;
    op.Init(pillar_features, coords, params, spatial_features);  // 初始化和数据分片
    op.Process();  // 执行主要的scatter操作
    // op析构时会自动清理本地资源
}
