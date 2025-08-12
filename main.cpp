/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <ctime>
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_pillar_scatter_custom.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void pillar_scatter_custom(GM_ADDR pillar_features, 
                                                            GM_ADDR coords, 
                                                            GM_ADDR params, 
                                                            GM_ADDR spatial_features);
#endif

// 获取文件大小的辅助函数
size_t getFileSize(const char* filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size;
    }
    return 0;
}

int32_t main(int32_t argc, char *argv[])
{
    // 设置并行块数，与算子实现保持一致
    uint32_t blockDim = 8;
    
    // PillarScatter参数配置
    constexpr int32_t PILLAR_FEATURE_SIZE = 64;
    constexpr int32_t FEATURE_X = 1024;
    constexpr int32_t FEATURE_Y = 1024;
    
    // 根据输入文件大小自动计算pillar数量
    const char* pillarFeaturesFile = "./input/OpTest_scatter_input_x.bin";
    const char* coordsFile = "./input/OpTest_scatter_input_coords.bin";
    
    size_t pillarFeaturesFileSize = getFileSize(pillarFeaturesFile);
    size_t coordsFileSize = getFileSize(coordsFile);
    
    if (pillarFeaturesFileSize == 0 || coordsFileSize == 0) {
        printf("错误：无法读取输入文件大小\n");
        return -1;
    }
    
    // 计算pillar数量
    // pillar_features: [num_pillars, 64] float16, 每个元素2字节
    uint32_t num_pillars_from_features = pillarFeaturesFileSize / (PILLAR_FEATURE_SIZE * sizeof(uint16_t));
    // coords: [num_pillars, 4] int32, 每个元素4字节
    uint32_t num_pillars_from_coords = coordsFileSize / (4 * sizeof(uint32_t));
    
    if (num_pillars_from_features != num_pillars_from_coords) {
        printf("警告：从特征文件和坐标文件计算出的pillar数量不一致！\n");
        printf("  特征文件推算：%u pillars\n", num_pillars_from_features);
        printf("  坐标文件推算：%u pillars\n", num_pillars_from_coords);
        printf("  使用较小值以避免越界\n");
    }
    
    uint32_t num_pillars = std::min(num_pillars_from_features, num_pillars_from_coords);
    printf("检测到输入数据包含 %u 个pillars\n", num_pillars);
    
    // 计算输入输出数据大小
    size_t pillarFeaturesSize = num_pillars * PILLAR_FEATURE_SIZE * sizeof(uint16_t);  // [N, 64] float16
    size_t coordsSize = num_pillars * 4 * sizeof(uint32_t) + 8 * sizeof(uint32_t);                           // [N, 4] int32 +8防止越界
    size_t paramsSize = 1 * sizeof(uint32_t);                                         // 存储pillar数量
    size_t spatialFeaturesSize = 1 * FEATURE_Y * FEATURE_X * PILLAR_FEATURE_SIZE * sizeof(uint16_t); // [1, 720, 720, 64] float16 (NHWC)

#ifdef ASCENDC_CPU_DEBUG
    // 在CPU调试模式下，分配主机内存用于输入输出
    uint8_t *pillarFeatures = (uint8_t *)AscendC::GmAlloc(pillarFeaturesSize);
    uint8_t *coords = (uint8_t *)AscendC::GmAlloc(coordsSize);
    uint8_t *params = (uint8_t *)AscendC::GmAlloc(paramsSize);
    uint8_t *spatialFeatures = (uint8_t *)AscendC::GmAlloc(spatialFeaturesSize);
    
    // 设置params参数（pillar数量）
    *((uint32_t*)params) = num_pillars;

    // 从文件读取输入数据到主机内存
    ReadFile(pillarFeaturesFile, pillarFeaturesSize, pillarFeatures, pillarFeaturesSize);
    ReadFile(coordsFile, coordsSize, coords, coordsSize);
    
    // 初始化输出内存为0
    memset(spatialFeatures, 0, spatialFeaturesSize);

    // 设置内核模式为AIV_MODE，适配昇腾C算子
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    // 开始计时
    printf("\n========== 算子执行时间统计 ==========\n");
    printf("开始执行PillarScatter算子 (CPU模式)...\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 获取高精度时间戳
    auto start_time_sys = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time_sys);
    auto start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(start_time_sys.time_since_epoch()) % 1000000;
    
    // 格式化输出时间戳（精确到微秒）
    struct tm* start_tm = localtime(&start_time_t);
    printf("开始时间: %04d-%02d-%02d %02d:%02d:%02d.%06ld\n",
           start_tm->tm_year + 1900, start_tm->tm_mon + 1, start_tm->tm_mday,
           start_tm->tm_hour, start_tm->tm_min, start_tm->tm_sec, start_time_us.count());
    
    // 在CPU上直接调用pillar_scatter_custom算子，blockDim为并行块数
    ICPU_RUN_KF(pillar_scatter_custom, blockDim, pillarFeatures, coords, params, spatialFeatures);
    
    // 结束计时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto end_time_sys = std::chrono::system_clock::now();
    auto end_time_t = std::chrono::system_clock::to_time_t(end_time_sys);
    auto end_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time_sys.time_since_epoch()) % 1000000;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 格式化输出结束时间戳（精确到微秒）
    struct tm* end_tm = localtime(&end_time_t);
    printf("结束时间: %04d-%02d-%02d %02d:%02d:%02d.%06ld\n",
           end_tm->tm_year + 1900, end_tm->tm_mon + 1, end_tm->tm_mday,
           end_tm->tm_hour, end_tm->tm_min, end_tm->tm_sec, end_time_us.count());
    
    // 显示精确的执行时间差（使用high_resolution_clock测量）
    printf("执行时间差: %ld μs (%.3f ms)\n", duration.count(), duration.count() / 1000.0);
    
    printf("✓ 算子执行完成!\n");
    printf("执行时间: %.3f ms (%.6f 秒)\n", 
           duration.count() / 1000.0, duration.count() / 1000000.0);
    printf("处理pillar数量: %u\n", num_pillars);
    printf("平均每个pillar处理时间: %.3f μs\n", (double)duration.count() / num_pillars);
    printf("吞吐量: %.2f K pillars/秒\n", num_pillars / (duration.count() / 1000000.0) / 1000.0);
    printf("=====================================\n\n");

    // 验证输出数据
    uint16_t* outputPtr = (uint16_t*)spatialFeatures;
    size_t totalElements = 1 * FEATURE_Y * FEATURE_X * PILLAR_FEATURE_SIZE;
    size_t nonZeroCount = 0;
    uint16_t firstNonZeroHalf = 0;
    size_t firstNonZeroIdx = 0;
    
    for (size_t i = 0; i < totalElements; i++) {
        if (outputPtr[i] != 0) {
            if (nonZeroCount == 0) {
                firstNonZeroHalf = outputPtr[i];
                firstNonZeroIdx = i;
            }
            nonZeroCount++;
        }
    }
    
    printf("\n输出数据验证 (NHWC格式):\n");
    printf("  总元素数: %zu\n", totalElements);
    printf("  非零元素数: %zu (%.2f%%)\n", nonZeroCount, (float)nonZeroCount / totalElements * 100);
    if (nonZeroCount > 0) {
        printf("  第一个非零值: 0x%04X (位置: %zu)\n", firstNonZeroHalf, firstNonZeroIdx);
        // 将位置转换为NHWC坐标
        size_t h = firstNonZeroIdx / (FEATURE_X * PILLAR_FEATURE_SIZE);
        size_t w = (firstNonZeroIdx % (FEATURE_X * PILLAR_FEATURE_SIZE)) / PILLAR_FEATURE_SIZE;
        size_t c = firstNonZeroIdx % PILLAR_FEATURE_SIZE;
        printf("  对应坐标: H=%zu, W=%zu, C=%zu\n", h, w, c);
    } else {
        printf("  警告：输出全是0！\n");
    }

    // 将输出结果写入文件
    WriteFile("./output/OpTest_scatter_output_x.bin", spatialFeatures, spatialFeaturesSize);

    // 释放主机内存
    AscendC::GmFree((void *)pillarFeatures);
    AscendC::GmFree((void *)coords);
    AscendC::GmFree((void *)params);
    AscendC::GmFree((void *)spatialFeatures);
#else
    // 初始化ACL环境
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    // 设置当前设备
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    // 创建ACL流，用于异步任务
    CHECK_ACL(aclrtCreateStream(&stream));

    // 分别为主机和设备分配输入输出内存
    uint8_t *pillarFeaturesHost, *coordsHost, *paramsHost, *spatialFeaturesHost;
    uint8_t *pillarFeaturesDevice, *coordsDevice, *paramsDevice, *spatialFeaturesDevice;

    // 分配主机内存
    CHECK_ACL(aclrtMallocHost((void **)(&pillarFeaturesHost), pillarFeaturesSize));
    CHECK_ACL(aclrtMallocHost((void **)(&coordsHost), coordsSize));
    CHECK_ACL(aclrtMallocHost((void **)(&paramsHost), paramsSize));
    CHECK_ACL(aclrtMallocHost((void **)(&spatialFeaturesHost), spatialFeaturesSize));
    
    // 分配设备内存
    CHECK_ACL(aclrtMalloc((void **)&pillarFeaturesDevice, pillarFeaturesSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&coordsDevice, coordsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&paramsDevice, paramsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&spatialFeaturesDevice, spatialFeaturesSize, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // 设置params参数（pillar数量）
    *((uint32_t*)paramsHost) = num_pillars;

    // 从文件读取输入数据到主机内存
    ReadFile(pillarFeaturesFile, pillarFeaturesSize, pillarFeaturesHost, pillarFeaturesSize);
    ReadFile(coordsFile, coordsSize, coordsHost, coordsSize);
    
    // 初始化输出内存为0
    memset(spatialFeaturesHost, 0, spatialFeaturesSize);

    // 将主机内存数据拷贝到设备内存
    CHECK_ACL(aclrtMemcpy(pillarFeaturesDevice, pillarFeaturesSize, pillarFeaturesHost, pillarFeaturesSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(coordsDevice, coordsSize, coordsHost, coordsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(paramsDevice, paramsSize, paramsHost, paramsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(spatialFeaturesDevice, spatialFeaturesSize, spatialFeaturesHost, spatialFeaturesSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // 开始计时
    printf("\n========== 算子执行时间统计 ==========\n");
    printf("开始执行PillarScatter算子 (NPU模式)...\n");
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 获取高精度时间戳
    auto start_time_sys = std::chrono::system_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(start_time_sys);
    auto start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(start_time_sys.time_since_epoch()) % 1000000;
    
    // 格式化输出时间戳（精确到微秒）
    struct tm* start_tm = localtime(&start_time_t);
    printf("开始时间: %04d-%02d-%02d %02d:%02d:%02d.%06ld\n",
           start_tm->tm_year + 1900, start_tm->tm_mon + 1, start_tm->tm_mday,
           start_tm->tm_hour, start_tm->tm_min, start_tm->tm_sec, start_time_us.count());

    // 启动自定义算子内核，blockDim为并行块数，stream为ACL流
    ACLRT_LAUNCH_KERNEL(pillar_scatter_custom)(blockDim, stream, pillarFeaturesDevice, coordsDevice, paramsDevice, spatialFeaturesDevice);
    
    // 等待流中所有任务完成，确保计算结束
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // 结束计时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto end_time_sys = std::chrono::system_clock::now();
    auto end_time_t = std::chrono::system_clock::to_time_t(end_time_sys);
    auto end_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time_sys.time_since_epoch()) % 1000000;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 格式化输出结束时间戳（精确到微秒）
    struct tm* end_tm = localtime(&end_time_t);
    printf("结束时间: %04d-%02d-%02d %02d:%02d:%02d.%06ld\n",
           end_tm->tm_year + 1900, end_tm->tm_mon + 1, end_tm->tm_mday,
           end_tm->tm_hour, end_tm->tm_min, end_tm->tm_sec, end_time_us.count());
    
    // 显示精确的执行时间差（使用high_resolution_clock测量）
    printf("执行时间差: %ld μs (%.3f ms)\n", duration.count(), duration.count() / 1000.0);
    
    printf("✓ 算子执行完成!\n");
    printf("执行时间: %.3f ms (%.6f 秒)\n", 
           duration.count() / 1000.0, duration.count() / 1000000.0);
    printf("处理pillar数量: %u\n", num_pillars);
    printf("平均每个pillar处理时间: %.3f μs\n", (double)duration.count() / num_pillars);
    printf("吞吐量: %.2f K pillars/秒\n", num_pillars / (duration.count() / 1000000.0) / 1000.0);
    printf("=====================================\n\n");

    // 将设备内存的输出数据拷贝回主机内存
    CHECK_ACL(aclrtMemcpy(spatialFeaturesHost, spatialFeaturesSize, spatialFeaturesDevice, spatialFeaturesSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // 验证输出数据
    uint16_t* outputPtr = (uint16_t*)spatialFeaturesHost;
    size_t totalElements = 1 * FEATURE_Y * FEATURE_X * PILLAR_FEATURE_SIZE;
    size_t nonZeroCount = 0;
    uint16_t firstNonZeroHalf = 0;
    size_t firstNonZeroIdx = 0;
    
    for (size_t i = 0; i < totalElements; i++) {
        if (outputPtr[i] != 0) {
            if (nonZeroCount == 0) {
                firstNonZeroHalf = outputPtr[i];
                firstNonZeroIdx = i;
            }
            nonZeroCount++;
        }
    }
    
    printf("\n输出数据验证 (NHWC格式):\n");
    printf("  总元素数: %zu\n", totalElements);
    printf("  非零元素数: %zu (%.2f%%)\n", nonZeroCount, (float)nonZeroCount / totalElements * 100);
    if (nonZeroCount > 0) {
        printf("  第一个非零值: 0x%04X (位置: %zu)\n", firstNonZeroHalf, firstNonZeroIdx);
        // 将位置转换为NHWC坐标
        size_t h = firstNonZeroIdx / (FEATURE_X * PILLAR_FEATURE_SIZE);
        size_t w = (firstNonZeroIdx % (FEATURE_X * PILLAR_FEATURE_SIZE)) / PILLAR_FEATURE_SIZE;
        size_t c = firstNonZeroIdx % PILLAR_FEATURE_SIZE;
        printf("  对应坐标: H=%zu, W=%zu, C=%zu\n", h, w, c);
    } else {
        printf("  警告：输出全是0！\n");
    }

    // 将输出结果写入文件
    WriteFile("./output/OpTest_scatter_output_x.bin", spatialFeaturesHost, spatialFeaturesSize);

    // 释放设备和主机内存
    CHECK_ACL(aclrtFree(pillarFeaturesDevice));
    CHECK_ACL(aclrtFree(coordsDevice));
    CHECK_ACL(aclrtFree(paramsDevice));
    CHECK_ACL(aclrtFree(spatialFeaturesDevice));
    CHECK_ACL(aclrtFreeHost(pillarFeaturesHost));
    CHECK_ACL(aclrtFreeHost(coordsHost));
    CHECK_ACL(aclrtFreeHost(paramsHost));
    CHECK_ACL(aclrtFreeHost(spatialFeaturesHost));

    // 销毁流，重置设备，反初始化ACL环境
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    // 程序正常结束
    return 0;
}
