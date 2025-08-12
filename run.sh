#!/bin/bash
# 获取当前脚本所在目录，并切换到该目录
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

# 设置默认构建类型和安装前缀
BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"

# 解析命令行参数，支持短参数和长参数
SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

RUN_MODE="npu"  # Set default RUN_MODE to npu
SOC_VERSION="Ascend310P1"
TOOLKIT_VERSION="8.0.RC2"  # Default toolkit version

# 处理命令行参数，设置运行模式、芯片型号等
while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

# 检查运行模式参数是否合法
RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "ERROR: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

# 检查芯片型号参数是否合法
VERSION_LIST="Ascend910A Ascend910B Ascend910ProA Ascend910ProB Ascend910PremiumA Ascend310B1 Ascend310B2 Ascend310B3 Ascend310B4 Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

# 自动检测或设置昇腾CANN工具包安装路径
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/${TOOLKIT_VERSION}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0:/usr/local/Ascend/ascend-toolkit/${TOOLKIT_VERSION}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu:$CPLUS_INCLUDE_PATH

# 设置环境变量，指向CANN工具包
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
# 根据运行模式配置环境变量和库路径
if [ "${RUN_MODE}" = "npu" ]; then
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
elif [ "${RUN_MODE}" = "sim" ]; then
    # 仿真模式下，使用stub库并设置仿真日志目录
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/runtime/lib64/stub:$LD_LIBRARY_PATH
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
    if [ ! $CAMODEL_LOG_PATH ]; then
        export CAMODEL_LOG_PATH=$(pwd)/sim_log
    fi
    if [ -d "$CAMODEL_LOG_PATH" ]; then
        rm -rf $CAMODEL_LOG_PATH
    fi
    mkdir -p $CAMODEL_LOG_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -e # 脚本遇到错误立即退出
# 清理旧的构建和输出目录
rm -rf build out
mkdir -p build
# 配置CMake工程，传递运行模式、芯片型号等参数
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
# 编译工程
cmake --build build -j
# 安装编译产物到out目录
cmake --install build

# 处理内核二进制文件
rm -f ascendc_kernels_bbit
cp ./out/bin/ascendc_kernels_bbit ./
# 不清理input和output目录，保留已有文件
# 只删除将要生成的输出文件，避免删除参考文件
echo "准备输出目录..."
mkdir -p output
if [ -f "./output/OpTest_scatter_output_x.bin" ]; then
    echo "删除旧的输出文件：OpTest_scatter_output_x.bin"
    rm -f ./output/OpTest_scatter_output_x.bin
fi

# 检查输入文件是否存在
if [ ! -d "input" ]; then
    echo "错误：input目录不存在，请创建并放入输入文件"
    exit 1
fi

# 注释掉数据生成，因为输入文件已经准备好
# python3 scripts/gen_data.py
echo "检查输入文件..."
if [ -f "./input/OpTest_scatter_input_x.bin" ] && [ -f "./input/OpTest_scatter_input_coords.bin" ]; then
    echo "✓ 输入文件已就绪："
    ls -la ./input/OpTest_scatter_input_x.bin
    ls -la ./input/OpTest_scatter_input_coords.bin
else
    echo "错误：缺少输入文件！"
    echo "需要以下文件："
    echo "  - ./input/OpTest_scatter_input_x.bin"
    echo "  - ./input/OpTest_scatter_input_coords.bin"
    exit 1
fi

echo "输出文件将生成在："
echo "  - ./output/OpTest_scatter_output_x.bin"

echo ""
echo "========================================="
echo "准备运行PillarScatter算子..."
echo "程序将自动检测输入数据中的pillar数量"
echo "========================================="
echo ""

(
    # 设置运行时库路径，确保可执行文件能找到依赖库
    export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
    # 判断是否用msprof工具链分析，否则直接运行主程序
    if [[ "$RUN_WITH_TOOLCHAIN" -eq 1 ]]; then
        if [ "${RUN_MODE}" = "npu" ]; then
            msprof op --application=./ascendc_kernels_bbit
        elif [ "${RUN_MODE}" = "sim" ]; then
            msprof op simulator --application=./ascendc_kernels_bbit
        elif [ "${RUN_MODE}" = "cpu" ]; then
            ./ascendc_kernels_bbit
        fi
    else
        ./ascendc_kernels_bbit
    fi
)

# 校验输出结果的md5值
if [ -f "./output/OpTest_scatter_output_x.bin" ]; then
    echo ""
    echo "✓ 输出文件已生成："
    ls -la ./output/OpTest_scatter_output_x.bin
    md5sum ./output/OpTest_scatter_output_x.bin
    
    # 如果存在参考文件，自动运行对比脚本
    if [ -f "./output/OpTest_scatter_output_x_correct.bin" ]; then
        echo ""
        echo "运行对比验证..."
        python3 check.py
    fi
else
    echo "错误：输出文件未生成"
fi

echo ""
echo "PillarScatter算子执行完成！"
