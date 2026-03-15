#!/bin/bash
# MoE Threshold 全量扫描脚本（带正确性校验和动态基准）
# 用法: ./scripts/bench-moe-threshold-full.sh

set -e

MODEL="$HOME/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q4_K_M.gguf"
RESULT_FILE=".solar/moe-threshold-benchmark-$(date +%Y%m%d-%H%M).md"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MoE Threshold Full Benchmark with Correctness Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 确保环境变量
export METAL_FUSION=1
export FUSED_QKV=1
export USE_MPS_GRAPH=0

# ===================================================================
# Phase 1: 生成 Golden Output Hash (threshold=32, baseline)
# ===================================================================
echo "Phase 1: Generating golden output hash (threshold=32)..."
export THUNDER_MOE_THRESHOLD=32

GOLDEN_OUTPUT=$(./build/bin/llama-cli \
    -m "$MODEL" \
    -p "Once upon a time" \
    -n 64 \
    --seed 42 \
    -ngl 99 \
    -fa 1 \
    2>/dev/null)

GOLDEN_HASH=$(echo "$GOLDEN_OUTPUT" | shasum -a 256 | awk '{print $1}')
echo "Golden Hash: $GOLDEN_HASH"
echo ""

# ===================================================================
# Phase 2: 动态测量 Baseline 性能 (threshold=32)
# ===================================================================
echo "Phase 2: Measuring baseline performance (threshold=32)..."
./scripts/bench-moe-threshold.sh

BASELINE_TG=$(grep "tg128" "/tmp/moe-threshold-32.log" | awk '{print $NF}' | cut -d± -f1 | xargs)
echo "Measured Baseline TG: $BASELINE_TG tok/s"
echo ""

# ===================================================================
# Phase 3: 初始化结果文件
# ===================================================================
cat > "$RESULT_FILE" <<EOF
# MoE Threshold Benchmark - Q4_K_M

**日期**: $(date +"%Y-%m-%d %H:%M:%S")
**模型**: Q4_K_M
**动态基准**: $BASELINE_TG tok/s (threshold=32)
**Golden Hash**: \`$GOLDEN_HASH\`

---

## 测试结果

| Threshold | TG (tok/s) | Delta vs Baseline | Correctness |
|-----------|------------|-------------------|-------------|
EOF

# ===================================================================
# Phase 4: 扫描所有阈值
# ===================================================================
echo "Phase 3: Scanning all thresholds..."
echo ""

for threshold in 32 16 8 4 2 1; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing THUNDER_MOE_THRESHOLD=$threshold"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    export THUNDER_MOE_THRESHOLD=$threshold

    # 正确性检查
    if [ "$threshold" != "32" ]; then
        echo "Correctness check..."
        CURRENT_OUTPUT=$(./build/bin/llama-cli \
            -m "$MODEL" \
            -p "Once upon a time" \
            -n 64 \
            --seed 42 \
            -ngl 99 \
            -fa 1 \
            2>/dev/null)

        CURRENT_HASH=$(echo "$CURRENT_OUTPUT" | shasum -a 256 | awk '{print $1}')

        if [ "$GOLDEN_HASH" = "$CURRENT_HASH" ]; then
            CORRECTNESS="✅ PASS"
        else
            CORRECTNESS="❌ FAIL"
            echo "⚠️  WARNING: Output mismatch!"
            echo "   Expected: $GOLDEN_HASH"
            echo "   Got:      $CURRENT_HASH"
        fi
    else
        CORRECTNESS="✅ PASS (baseline)"
    fi

    # 性能测试
    ./scripts/bench-moe-threshold.sh

    # 提取结果并计算 delta
    TG=$(grep "tg128" "/tmp/moe-threshold-$threshold.log" | awk '{print $NF}' | cut -d± -f1 | xargs)
    DELTA=$(echo "scale=2; (($TG - $BASELINE_TG) / $BASELINE_TG) * 100" | bc)

    echo "| $threshold | $TG | ${DELTA}% | $CORRECTNESS |" >> "$RESULT_FILE"

    echo ""
done

# ===================================================================
# Phase 5: 总结
# ===================================================================
cat >> "$RESULT_FILE" <<EOF

---

## 分析

**最优阈值**: （根据上表手动标注）

**建议**:
- 如果所有阈值都通过正确性检查且有性能提升 ≥ 5%，选择提升最大的阈值
- 如果有阈值未通过正确性检查（❌ FAIL），**禁止使用**
- 如果所有低阈值都降低性能，保持默认 32

---

*生成时间: $(date +"%Y-%m-%d %H:%M:%S")*
*工具: bench-moe-threshold-full.sh*
EOF

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results saved to: $RESULT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat "$RESULT_FILE"
