#!/bin/bash
# MoE Threshold Benchmark Script - 单次测试
# 用法: THUNDER_MOE_THRESHOLD=X ./scripts/bench-moe-threshold.sh

set -e

THRESHOLD=${THUNDER_MOE_THRESHOLD:-32}
MODEL="$HOME/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q4_K_M.gguf"
EXPECTED_TG=79.12

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MoE Threshold Benchmark - THRESHOLD=$THRESHOLD"
echo "Model: Q4_K_M"
echo "Expected baseline: $EXPECTED_TG tok/s"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 确保环境变量
export METAL_FUSION=1
export FUSED_QKV=1
export USE_MPS_GRAPH=0
export THUNDER_MOE_THRESHOLD=$THRESHOLD

echo ""
echo "Environment Variables:"
echo "  METAL_FUSION=$METAL_FUSION"
echo "  FUSED_QKV=$FUSED_QKV"
echo "  USE_MPS_GRAPH=$USE_MPS_GRAPH"
echo "  THUNDER_MOE_THRESHOLD=$THUNDER_MOE_THRESHOLD"
echo ""

# 执行性能测试
echo "Running performance benchmark..."
./build/bin/llama-bench \
    -m "$MODEL" \
    -t 4 -ngl 99 -r 5 -p 0 -n 128 -fa 1 \
    2>&1 | tee "/tmp/moe-threshold-$THRESHOLD.log"

# 提取结果
TG=$(grep "tg128" "/tmp/moe-threshold-$THRESHOLD.log" | awk '{print $NF}' | cut -d± -f1 | xargs)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Result: TG = $TG tok/s (threshold=$THRESHOLD)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
