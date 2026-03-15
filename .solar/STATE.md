# Mission
ThunderLLAMA 持续优化:
1. **L2/L3 两层缓存系统** -- CPU (2GB) + Disk (100GB) 缓存，98.5% 命中率
2. **Metal GPU 深度优化** -- Tier A/B 内核融合与 GEMV 加速 (Q5_K 65->72+ tok/s)
3. **Split-K Decode GEMV** -- 通过 K 维度并行提升带宽利用率 (目标 85-105 tok/s)

# Constraints
- 不破坏现有 Clawgate + ThunderLLAMA 集成
- 配置文件 (thunderllama.conf) 是唯一真相源
- 正确性优先：输出必须与未优化版本 bit-identical
- **禁止 Mock 和模拟**：代码必须真实实现，能跑通

# Current Plan

## #1 MPS 集成 — Phase 1 完成，Phase 2 进行中

### Phase 1: MPS Foundation + FP16 GEMM (DONE)
- MPSMatrixMultiplication wrapper (ggml-metal-mps.h/.mm)
- CMake 集成 MetalPerformanceShaders framework
- Device context 扩展 (mps_ctx)
- ggml-metal-ops.cpp MUL_MAT dispatch 分支
- Encoder pause/resume 机制
- thunderllama.conf USE_MPS_GRAPH 配置项
- 编译验证：0 errors, 0 warnings

### Phase 2: Split-K Decode GEMV — Q4_0 PoC NO-GO

**关键发现**：
- MPSGraph **没有** `quantizedMatmul` API（那是 MLX 的）
- MLX 使用自定义 Metal kernel (STEEL GEMM)，不是 MPSGraph
- 方案A（dequant→MPS matmul）因 Prefill 已被 LMCache 缓存而淘汰
- **方案B (Split-K fused kernel)**：Q4_0 PoC 已实现，**No-Go**

**Q4_0 Split-K PoC 结果 (2026-03-15)**：
| Config | Q4_0 TG (tok/s) | Delta |
|--------|-----------------|-------|
| Baseline (USE_MPS_GRAPH=0) | 73.26 ± 0.55 | — |
| Split-K NSG_K=4 | 72.50 ± 0.41 | **-1.0%** |

**根因分析**：
- Q4_0 带宽利用率已达 ~77%，留给 Split-K 的空间有限
- Intra-TG Split-K 不增加总内存并行度：Original 16 TG/core × 2 SG = 32 SG ≈ Split-K 4 TG/core × 8 SG = 32 SG
- Shared memory barrier + reduction 开销抵消了微弱的收益
- 结论：Q4_0 kernel 已接近带宽上限，intra-TG Split-K 无法突破

**待决策**：是否继续 Q5_K Split-K（BW util ~53%，可能有更大空间），或转向其他优化方向

---

## LMCache 两层缓存系统

### 当前架构：L2 (CPU) + L3 (Disk)
- **L2 CPU 缓存**: 2GB RAM
- **L3 磁盘缓存**: 100GB Disk (`/Volumes/toshiba/thunderllama-cache/kv_cache.bin`)
- **当前性能**: L2 命中率 98.5%, Full Skip 20% 跳过率

### L1 GPU Pool 已移除 (2026-03-15)
因 llama-server slot 管理清空 KV cache，导致缓存失效。

---

## Metal GPU 内核优化

## Tier A: 高回报、可行性高
| # | 优化方案 | 预估提升 | 状态 |
|---|---------|---------|------|
| A1 | Fused Expert Aggregation (7xADD -> 1 kernel) | +7.3% TG (实测) | 已完成 |
| A2 | Q5_K Branchless Dequant | +2-5% TG | 待做 |
| A3 | MoE ne21_mm_id_min 阈值降低 | +1.7% TG (实测) | ✅ 已完成 |
| A4 | Fused RMS_NORM+MUL+SWIGLU | +5-10% | 待做 |

## Tier B: 中等回报、技术挑战大
| # | 优化方案 | 预估提升 | 状态 |
|---|---------|---------|------|
| B1 | **Split-K Quantized GEMV** | **+25-45% TG** | **Q4_0 No-Go (-1.0%), 待决策** |
| B2 | MoE Expert-Only Dispatch | +5-15% | 待做 |
| B3 | MoE map0 Barrier 消除 | +3-8% | 待做 |

---

## 独立任务
- [x] **#1 MPS 集成 Phase 1** -- MPS 基础设施 + FP16 GEMM PoC
- [ ] **#1 MPS 集成 Phase 2** -- Split-K decode GEMV 优化
- [ ] **#2 Paged Attention 优化** -- 对标 vllm-mlx
- [x] **#9 LRU 策略优化** -- LMCache 生产级升级

## Metal JIT Fusion Pipeline (Week 6-8)
- [ ] #3 Week 6.1: ggml Graph 分析器
- [ ] #4 Week 6.2: Dependency Graph Builder (<- #3)
- [ ] #5 Week 7.1: Graph Rewriter (<- #3, #4)
- [ ] #6 Week 7.2: Metal Kernel Code Generator (<- #5)
- [ ] #7 Week 8.1: Metal JIT Compiler (<- #6)
- [ ] #8 Week 8.2: 集成到 ggml-metal (<- #7)

# Decisions

## MPS 集成决策
- [2026-03-15] **Phase 1 选择 MPSMatrixMultiplication 而非 MPSGraph**：MPSMatrixMultiplication 原生支持 buffer offset，更适合 ggml 的内存布局
- [2026-03-15] **Phase 2 淘汰方案A (MPS GEMM for prefill)**：Prefill 已被 LMCache 缓存加速，不是热点。Decode GEMV (BS=1) 是瓶颈
- [2026-03-15] **Phase 2 选定方案B (Split-K fused kernel)**：不重写 dequant 逻辑，核心改动是将 K 维度拆分到多个 threadgroup
- [2026-03-15] **MPSGraph 没有 quantizedMatmul**：原计划假设错误。该 API 属于 MLX，不属于 Apple MPSGraph
- [2026-03-15] **带宽分析**：现有 kernel 4 路 K 并行，带宽利用率仅 50-55%，理论空间大
- [2026-03-15] **Split-K Q4_0 PoC No-Go**：-1.0% TG (73.26→72.50)。根因：Q4_0 BW util 已达 ~77%，intra-TG Split-K 不增加总内存并行度 (32 SG/core 不变)

## LMCache 架构决策
- [2026-03-15] 移除 L1 GPU Pool：slot 管理清空 KV cache 导致缓存失效
- [2026-03-15] KV Cache 量化选择 q8_0：-2.2% 性能，-50% 内存
- [2026-03-14] EXCLUSIVE 模式：禁用内置 prompt cache

## Metal 优化决策
- [2026-03-15] **A3 MoE Threshold=16**：+1.7% TG (80.61 tok/s)。阈值 ≤8 灾难性下降 (-45%~-72%)
- [2026-03-15] A1 验证：ADD 链融合已覆盖 MoE 聚合 (+7.3%)
- [2026-03-15] K/V Projection Fusion：+9.8% TG, 输出 IDENTICAL
- [2026-03-14] N_R0_Q5_K=8 编译时常量：7 组实测确认

# Progress

## Done

### MPS 集成 Phase 1 (2026-03-15)
- ggml-metal-mps.h/mm: MPSMatrixMultiplication wrapper
- CMakeLists.txt: MetalPerformanceShaders framework + .mm 源文件
- ggml-metal-device.h/m: mps_ctx 指针 + init/free/getter
- ggml-metal-ops.cpp: MPS dispatch 分支 + encoder pause/resume
- config-parser.h: USE_MPS_GRAPH 配置映射
- thunderllama.conf: USE_MPS_GRAPH=0
- 编译：0 errors, 0 warnings
- 服务器启动：MPS init 消息确认

### MPS Phase 2 研究 (2026-03-15)
- 确认 MPSGraph 没有 quantizedMatmul (属于 MLX)
- 分析 MLX STEEL GEMM 架构 (QuantizedBlockLoader, qdot, Split-K)
- 分析 ggml Q4_K/Q5_K block format (176 bytes / 256 elements)
- 分析现有 BS=1 kernel 带宽利用率 (~50-55%)
- 确定优化方向：Split-K decode GEMV

### 此前完成的优化
- K/V Projection Fusion: +9.8% TG, +8.5% PP
- Metal MoE Kernel Fusion: 3-op fusion, 96 dispatches eliminated
- KV Cache 量化: q8_0 推荐 (-2.2% 性能, -50% 内存)
- L2/L3 缓存: 98.5% L2 命中率
- LMCache 生产级升级: freq-protected LRU, TTL, warm API

## In-Progress
- Track 2: Q4_K 带宽 Profiling（MPS Phase 2 决策）

## Blocked
- Normalization chain fusion (graph scheduler 限制)

## Done (新增 2026-03-15)

### ✅ Tier A3: MoE 阈值优化（已完成）
**优化**: ne21_mm_id_min 从 32 降低到 16
**性能**: +1.7% TG (79.25 → 80.61 tok/s)
**方法**: 快速扫描 (1/8/16/32)，发现阈值 ≤8 灾难性下降
**代码**:
- `ggml-metal-ops.cpp`: 环境变量支持 + 安全解析 (strtol)
- `thunderllama.conf`: THUNDER_MOE_THRESHOLD=16
**报告**: `.solar/moe-threshold-benchmark-20260315.md`

## Done (2026-03-15 早期)
### 🚨 8.4% 性能回退调查（已解决）
**问题**: Q4_K TG 79.12 → 72.51 tok/s (-8.4%)
**根因**: llama-bench 测试时缺少 METAL_FUSION=1 环境变量，导致 MoE Kernel Fusion 被禁用
**验证**:
- 加上 METAL_FUSION=1: 79.81 ± 0.15 tok/s (恢复正常)
- 稳定性测试 3-run: 79.64/79.54/79.76 tok/s
**解决方案**:
- thunderllama.conf 中 METAL_FUSION=1 已设置（line 103）
- config-parser.h 正确映射到 GGML_METAL_FUSION_DISABLE
- 使用 llama-server 或启动脚本会自动读取配置
- 直接用 llama-bench 需手动设置环境变量

# Next Actions
1. **提交 Tier A3 代码** — 固化成果（ggml-metal-ops.cpp + thunderllama.conf + 测试报告）
2. **Track 2: Q4_K 带宽 Profiling** — 验证 BW util 假设，决定是否继续 Split-K
3. **基于 profiling 结果决策** — 继续 Split-K 或转向 Tier A (A2/A4)
