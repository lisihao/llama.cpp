#!/bin/bash
# ThunderLLAMA 标准性能测试脚本
# 创建于: 2026-03-15
# 用途: 所有性能对比测试必须使用此脚本，确保参数一致性

set -euo pipefail

# ============================================================================
# 配置区域 - 所有测试参数的唯一真相源
# ============================================================================

# 模型路径
MODEL="${MODEL:-$HOME/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q4_K_M.gguf}"

# 标准环境变量（与 thunderllama.conf 一致）
export METAL_FUSION=1
export FUSED_QKV=1
export USE_MPS_GRAPH=0          # 禁用 MPS Graph（已验证对性能有负面影响）
export THUNDER_LMCACHE=0        # 禁用 LMCache（测试基线性能）
export LLAMA_PAGED_ATTENTION=0  # 禁用 Paged Attention（测试基线性能）

# 标准测试参数（经过 M4 Pro 优化验证）
# -t 4: M4 Pro 最优 CPU 线程数（避免 GPU 带宽竞争）
# -fa 1: 启用 Flash Attention
# -ngl 99: 全部层 GPU offload
# -p 512: Prompt 处理 512 tokens
# -n 128: 生成 128 tokens
# -r 5: 5 次重复取平均（减少测量误差）
BENCH_PARAMS="-m $MODEL -p 512 -n 128 -ngl 99 -t 4 -fa 1 -r 5"

# ============================================================================
# 函数定义
# ============================================================================

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_config() {
    echo "环境变量:"
    echo "  METAL_FUSION=$METAL_FUSION"
    echo "  FUSED_QKV=$FUSED_QKV"
    echo "  USE_MPS_GRAPH=$USE_MPS_GRAPH"
    echo "  THUNDER_LMCACHE=$THUNDER_LMCACHE"
    echo "  LLAMA_PAGED_ATTENTION=$LLAMA_PAGED_ATTENTION"
    echo ""
    echo "测试参数: -t 4 -fa 1 -ngl 99 -p 512 -n 128 -r 5"
    echo ""
}

extract_tg_performance() {
    local log_file="$1"
    grep "tg128" "$log_file" | tail -1 | awk '{print $(NF-2)}'
}

extract_pp_performance() {
    local log_file="$1"
    grep "pp512" "$log_file" | tail -1 | awk '{print $(NF-2)}'
}

# ============================================================================
# 使用说明
# ============================================================================

show_usage() {
    cat <<EOF
用法: $0 <命令> [选项]

命令:
  baseline              运行基线测试（保存到 /tmp/benchmark-baseline.log）
  compare <描述>        运行对比测试（保存到 /tmp/benchmark-compare-<描述>.log）
  diff <baseline> <compare>  对比两个测试结果

示例:
  # 测试基线
  $0 baseline

  # 测试优化后版本
  $0 compare vectorized

  # 对比结果
  $0 diff /tmp/benchmark-baseline.log /tmp/benchmark-compare-vectorized.log

环境变量（可选覆盖）:
  MODEL=<path>          指定模型路径（默认: qwen3-30b Q4_K_M）

注意:
  - 所有测试使用相同的标准参数（-t 4 -fa 1 等）
  - 环境变量固定（METAL_FUSION=1, FUSED_QKV=1, 等）
  - 修改参数前请更新此脚本，确保所有测试一致
EOF
    exit 1
}

# ============================================================================
# 主命令
# ============================================================================

case "${1:-}" in
    baseline)
        print_header "运行基线测试"
        print_config

        OUTPUT_LOG="/tmp/benchmark-baseline.log"

        ./build/bin/llama-bench $BENCH_PARAMS 2>&1 | tee "$OUTPUT_LOG"

        echo ""
        echo "基线测试完成，结果保存到: $OUTPUT_LOG"
        echo ""
        echo "TG 性能:"
        grep "tg128" "$OUTPUT_LOG" | tail -1
        echo ""
        echo "PP 性能:"
        grep "pp512" "$OUTPUT_LOG" | tail -1
        ;;

    compare)
        if [ -z "${2:-}" ]; then
            echo "错误: 需要提供对比测试描述"
            echo "示例: $0 compare vectorized"
            exit 1
        fi

        DESCRIPTION="$2"
        print_header "运行对比测试: $DESCRIPTION"
        print_config

        OUTPUT_LOG="/tmp/benchmark-compare-$DESCRIPTION.log"

        ./build/bin/llama-bench $BENCH_PARAMS 2>&1 | tee "$OUTPUT_LOG"

        echo ""
        echo "对比测试完成，结果保存到: $OUTPUT_LOG"
        echo ""
        echo "TG 性能:"
        grep "tg128" "$OUTPUT_LOG" | tail -1
        echo ""
        echo "PP 性能:"
        grep "pp512" "$OUTPUT_LOG" | tail -1
        ;;

    diff)
        if [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
            echo "错误: 需要提供两个日志文件路径"
            echo "示例: $0 diff /tmp/baseline.log /tmp/compare.log"
            exit 1
        fi

        BASELINE_LOG="$2"
        COMPARE_LOG="$3"

        if [ ! -f "$BASELINE_LOG" ]; then
            echo "错误: 基线日志文件不存在: $BASELINE_LOG"
            exit 1
        fi

        if [ ! -f "$COMPARE_LOG" ]; then
            echo "错误: 对比日志文件不存在: $COMPARE_LOG"
            exit 1
        fi

        print_header "性能对比"

        BASELINE_TG=$(extract_tg_performance "$BASELINE_LOG")
        COMPARE_TG=$(extract_tg_performance "$COMPARE_LOG")

        BASELINE_PP=$(extract_pp_performance "$BASELINE_LOG")
        COMPARE_PP=$(extract_pp_performance "$COMPARE_LOG")

        echo "基线 ($(basename "$BASELINE_LOG")):"
        echo "  TG128: $BASELINE_TG tok/s"
        echo "  PP512: $BASELINE_PP tok/s"
        echo ""
        echo "对比 ($(basename "$COMPARE_LOG")):"
        echo "  TG128: $COMPARE_TG tok/s"
        echo "  PP512: $COMPARE_PP tok/s"
        echo ""

        if [ -n "$BASELINE_TG" ] && [ -n "$COMPARE_TG" ]; then
            DELTA_TG=$(echo "scale=2; $COMPARE_TG - $BASELINE_TG" | bc)
            PERCENT_TG=$(echo "scale=2; ($COMPARE_TG - $BASELINE_TG) / $BASELINE_TG * 100" | bc)
            echo "TG 变化: $DELTA_TG tok/s ($PERCENT_TG%)"
        fi

        if [ -n "$BASELINE_PP" ] && [ -n "$COMPARE_PP" ]; then
            DELTA_PP=$(echo "scale=2; $COMPARE_PP - $BASELINE_PP" | bc)
            PERCENT_PP=$(echo "scale=2; ($COMPARE_PP - $BASELINE_PP) / $BASELINE_PP * 100" | bc)
            echo "PP 变化: $DELTA_PP tok/s ($PERCENT_PP%)"
        fi
        ;;

    *)
        show_usage
        ;;
esac
