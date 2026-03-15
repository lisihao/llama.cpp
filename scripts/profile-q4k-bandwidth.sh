#!/bin/bash
# Q4_K Bandwidth Profiling Script
# 用于 Instruments Metal System Trace 分析

set -e

MODEL="$HOME/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q4_K_M.gguf"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Q4_K Bandwidth Profiling"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "准备启动 llama-bench 进行 profiling..."
echo ""
echo "请在另一个终端执行以下步骤："
echo ""
echo "1. 打开 Instruments:"
echo "   open -a Instruments"
echo ""
echo "2. 选择 'Metal System Trace' 模板"
echo ""
echo "3. 点击 Record (红色按钮)"
echo ""
echo "4. 回到本终端，按 Enter 启动测试..."
read -p ""

echo ""
echo "启动 llama-bench (固定 workload)..."
echo ""

export METAL_FUSION=1
export FUSED_QKV=1
export USE_MPS_GRAPH=0
export THUNDER_MOE_THRESHOLD=16

# 固定 workload：单次运行，128 tokens
./build/bin/llama-bench \
    -m "$MODEL" \
    -t 4 \
    -ngl 99 \
    -r 1 \
    -p 0 \
    -n 128 \
    -fa 1

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试完成！请在 Instruments 中停止录制。"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "分析要点："
echo "1. 查看 'Device Utilization' → GPU Memory Bandwidth %"
echo "2. 查找 'kernel_mul_mv_q4_K_f32' kernel"
echo "3. 记录 Bandwidth Utilization (目标: 验证是否 ~53%)"
echo "4. 查看 Limiter (Memory vs Compute)"
echo ""
