#!/bin/bash
# ThunderLLAMA Q4_K_M Benchmark Script
# 自动读取 thunderllama.conf 环境变量

set -e

# 读取配置文件
source <(grep -E "^[A-Z_]+=.*" thunderllama.conf | sed 's/^/export /')

# 展开 $HOME
MODEL_PATH=$(eval echo $MODEL_PATH)

# Benchmark 参数
THREADS=${1:-4}
RUNS=${2:-5}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ThunderLLAMA Q4_K_M Benchmark"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model:        $MODEL_PATH"
echo "Threads:      $THREADS"
echo "Runs:         $RUNS"
echo "METAL_FUSION: $METAL_FUSION"
echo "FUSED_QKV:    $FUSED_QKV"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

./build/bin/llama-bench \
    -m "$MODEL_PATH" \
    -t $THREADS \
    -ngl 99 \
    -r $RUNS \
    -p 0 \
    -n 128 \
    -fa 1 \
    2>&1 | grep -E "model|tg128|pp512"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
