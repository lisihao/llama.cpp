#!/bin/bash
# 标准化 Benchmark 脚本 - 防止遗漏环境变量
# 用法: ./bench-standard.sh [model] [threads] [runs]

set -e

# 默认参数
MODEL="${1:-~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q4_K_M.gguf}"
THREADS="${2:-4}"
RUNS="${3:-5}"

# 展开 ~ 路径
MODEL=$(eval echo "$MODEL")

# ===================================================================
# 强制检查：确保关键环境变量已设置
# ===================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "环境变量检查:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 读取配置文件设置
if [ -f "thunderllama.conf" ]; then
    METAL_FUSION_CONF=$(grep "^METAL_FUSION=" thunderllama.conf | cut -d= -f2 | tr -d ' ')
    FUSED_QKV_CONF=$(grep "^FUSED_QKV=" thunderllama.conf | cut -d= -f2 | tr -d ' ')
    echo "配置文件 (thunderllama.conf):"
    echo "  METAL_FUSION=$METAL_FUSION_CONF"
    echo "  FUSED_QKV=$FUSED_QKV_CONF"
else
    echo "⚠️  警告: thunderllama.conf 不存在"
    METAL_FUSION_CONF=1
    FUSED_QKV_CONF=1
fi

# 设置环境变量（覆盖任何现有设置）
export METAL_FUSION=${METAL_FUSION_CONF:-1}
export FUSED_QKV=${FUSED_QKV_CONF:-1}
export USE_MPS_GRAPH=0

echo ""
echo "实际使用的环境变量:"
echo "  METAL_FUSION=$METAL_FUSION"
echo "  FUSED_QKV=$FUSED_QKV"
echo "  USE_MPS_GRAPH=$USE_MPS_GRAPH"

# ===================================================================
# 验证：检查是否与记录基准一致
# ===================================================================
EXPECTED_TG=79.12
TOLERANCE=2.0  # ±2% 容忍度

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Benchmark 参数:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  模型:    $MODEL"
echo "  线程:    $THREADS"
echo "  轮数:    $RUNS"
echo "  预期 TG: $EXPECTED_TG ± ${TOLERANCE}%"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ===================================================================
# 执行 Benchmark
# ===================================================================
./build/bin/llama-bench \
    -m "$MODEL" \
    -t $THREADS \
    -ngl 99 \
    -r $RUNS \
    -p 0 \
    -n 128 \
    -fa 1 \
    2>&1 | tee /tmp/bench-result.txt

# ===================================================================
# 自动验证结果
# ===================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "结果验证:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TG_RESULT=$(grep "tg128" /tmp/bench-result.txt | awk '{print $NF}' | cut -d± -f1 | xargs)

if [ -n "$TG_RESULT" ]; then
    echo "  实测 TG: $TG_RESULT tok/s"
    echo "  基准 TG: $EXPECTED_TG tok/s"

    # 计算差异百分比
    DIFF=$(echo "scale=2; (($TG_RESULT - $EXPECTED_TG) / $EXPECTED_TG) * 100" | bc)
    ABS_DIFF=$(echo "scale=2; if ($DIFF < 0) -$DIFF else $DIFF" | bc)

    echo "  差异:    ${DIFF}%"

    # 判断是否在容忍范围内
    if (( $(echo "$ABS_DIFF < $TOLERANCE" | bc -l) )); then
        echo "  ✅ 性能正常 (在 ±${TOLERANCE}% 范围内)"
    else
        echo "  ⚠️  性能异常 (超出 ±${TOLERANCE}% 范围)"
        echo ""
        echo "可能的原因:"
        echo "  1. METAL_FUSION 未正确设置"
        echo "  2. 系统负载过高"
        echo "  3. 内存不足"
        echo "  4. 代码回归"
        echo ""
        echo "建议检查:"
        echo "  echo \$METAL_FUSION"
        echo "  echo \$FUSED_QKV"
        echo "  top -l 1 | grep PhysMem"
    fi
else
    echo "  ❌ 无法提取 TG 结果"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
