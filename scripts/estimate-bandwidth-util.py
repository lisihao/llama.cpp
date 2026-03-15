#!/usr/bin/env python3
"""
Q4_K Bandwidth Utilization Estimator
基于实测性能反推带宽利用率
"""

# 模型参数
MODEL_PARAMS = 30.53e9  # 30.53B parameters
HIDDEN_DIM = 3584       # Qwen3-30B hidden dimension
N_LAYERS = 48           # Number of layers
N_EXPERTS = 60          # Total experts
N_ACTIVE_EXPERTS = 2    # Active experts per token

# Q4_K 量化格式
# 每个 block: 256 elements = 176 bytes
# 每个 element: 176/256 = 0.6875 bytes
BYTES_PER_PARAM_Q4K = 176 / 256

# 实测性能
TG_THROUGHPUT = 80.61  # tokens/sec (实测)

# GPU 规格 (M4 Pro)
GPU_BANDWIDTH_GB_S = 273  # GB/s (理论峰值)

def estimate_memory_traffic_per_token():
    """
    估算每个 token decode 的内存流量
    """

    # 1. Attention 部分（Self-Attention）
    # Q/K/V 权重: 3 × (hidden_dim × hidden_dim) × layers
    qkv_params = 3 * HIDDEN_DIM * HIDDEN_DIM * N_LAYERS
    qkv_bytes = qkv_params * BYTES_PER_PARAM_Q4K

    # Attention output projection
    attn_out_params = HIDDEN_DIM * HIDDEN_DIM * N_LAYERS
    attn_out_bytes = attn_out_params * BYTES_PER_PARAM_Q4K

    # 2. MoE FFN 部分
    # 每层激活 2 个 expert，每个 expert 的参数量
    expert_params_per_layer = 2 * (HIDDEN_DIM * 4 * HIDDEN_DIM)  # 简化估算
    moe_bytes = expert_params_per_layer * N_LAYERS * BYTES_PER_PARAM_Q4K

    # 3. Gating network (小，忽略)

    # 4. Layer Norm 等（小，忽略）

    total_bytes = qkv_bytes + attn_out_bytes + moe_bytes

    return {
        'qkv_gb': qkv_bytes / 1e9,
        'attn_out_gb': attn_out_bytes / 1e9,
        'moe_gb': moe_bytes / 1e9,
        'total_gb': total_bytes / 1e9
    }

def estimate_bandwidth_utilization():
    """
    估算带宽利用率
    """

    traffic = estimate_memory_traffic_per_token()

    # 实际带宽需求
    actual_bandwidth_gb_s = traffic['total_gb'] * TG_THROUGHPUT

    # 带宽利用率
    bandwidth_util = (actual_bandwidth_gb_s / GPU_BANDWIDTH_GB_S) * 100

    return {
        'traffic_per_token_gb': traffic['total_gb'],
        'actual_bandwidth_gb_s': actual_bandwidth_gb_s,
        'theoretical_bandwidth_gb_s': GPU_BANDWIDTH_GB_S,
        'bandwidth_utilization_pct': bandwidth_util,
        'breakdown': traffic
    }

if __name__ == "__main__":
    print("=" * 60)
    print("Q4_K Bandwidth Utilization Estimation")
    print("=" * 60)
    print()

    print("Model Configuration:")
    print(f"  Parameters: {MODEL_PARAMS/1e9:.2f}B")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Layers: {N_LAYERS}")
    print(f"  MoE: {N_ACTIVE_EXPERTS}/{N_EXPERTS} experts")
    print(f"  Quantization: Q4_K ({BYTES_PER_PARAM_Q4K:.4f} bytes/param)")
    print()

    print("Performance:")
    print(f"  TG Throughput: {TG_THROUGHPUT:.2f} tokens/sec")
    print()

    print("GPU Specs (M4 Pro):")
    print(f"  Memory Bandwidth: {GPU_BANDWIDTH_GB_S} GB/s")
    print()

    result = estimate_bandwidth_utilization()

    print("Estimated Memory Traffic per Token:")
    print(f"  QKV Weights:     {result['breakdown']['qkv_gb']:.4f} GB")
    print(f"  Attn Output:     {result['breakdown']['attn_out_gb']:.4f} GB")
    print(f"  MoE FFN:         {result['breakdown']['moe_gb']:.4f} GB")
    print(f"  Total:           {result['traffic_per_token_gb']:.4f} GB")
    print()

    print("Bandwidth Utilization:")
    print(f"  Actual Bandwidth:   {result['actual_bandwidth_gb_s']:.2f} GB/s")
    print(f"  Theoretical Peak:   {result['theoretical_bandwidth_gb_s']:.2f} GB/s")
    print(f"  Utilization:        {result['bandwidth_utilization_pct']:.1f}%")
    print()

    print("=" * 60)
    print("Analysis:")
    print("=" * 60)

    if result['bandwidth_utilization_pct'] < 40:
        print("✅ LOW bandwidth utilization (<40%)")
        print("   → Split-K has LARGE potential (+50%+ possible)")
        print("   → Recommend: Continue Split-K implementation")
    elif result['bandwidth_utilization_pct'] < 70:
        print("⚠️  MEDIUM bandwidth utilization (40-70%)")
        print("   → Split-K has MODERATE potential (+10-30% possible)")
        print("   → Recommend: Prototype Split-K, validate with real profiling")
    else:
        print("❌ HIGH bandwidth utilization (>70%)")
        print("   → Split-K has LIMITED potential (<10%)")
        print("   → Recommend: Focus on Tier A optimizations (A2/A4)")

    print()
    print("Note: This is a THEORETICAL estimate. Real profiling with")
    print("Instruments Metal System Trace is needed for accurate data.")
    print()
