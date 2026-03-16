# Benchmark 测试检查清单

**目的**: 防止遗漏关键环境变量，确保测试结果可靠

---

## 📋 测试前检查清单（必须全部勾选）

### 环境变量检查

- [ ] `METAL_FUSION=1` 已设置（检查：`echo $METAL_FUSION`）
- [ ] `FUSED_QKV=1` 已设置（检查：`echo $FUSED_QKV`）
- [ ] `USE_MPS_GRAPH=0` 已设置（MPS Phase 2 完成前）

### 系统状态检查

- [ ] GPU 可用（检查：`system_profiler SPDisplaysDataType | grep Metal`）
- [ ] 内存充足（检查：`top -l 1 | grep PhysMem`，至少 10GB 可用）
- [ ] 无其他负载（检查：`ps aux | grep llama-server`）

### 配置一致性检查

- [ ] 测试命令与 thunderllama.conf 一致
  - 线程数：4 (TG) / 8 (PP)
  - GPU 层数：99
  - Flash Attention：启用 (-fa 1)

### 基准对比

- [ ] 记录基准性能（Q4_K: 79.12 tok/s, Q5_K: 65.25 tok/s）
- [ ] 测试轮数 ≥ 5（减少随机波动）
- [ ] 结果在基准 ±2% 范围内

---

## ✅ 推荐测试流程

### 方式 1: 使用标准化脚本（强烈推荐）⭐

```bash
# 位置: scripts/benchmark-standard.sh
cd /Users/lisihao/ThunderLLAMA

# 运行基线测试
./scripts/benchmark-standard.sh baseline

# 运行对比测试（例如测试某个优化）
./scripts/benchmark-standard.sh compare vectorized

# 对比结果
./scripts/benchmark-standard.sh diff /tmp/benchmark-baseline.log /tmp/benchmark-compare-vectorized.log
```

**优势**:
- ✅ 自动设置所有环境变量
- ✅ 强制使用正确参数 (-t 4 -fa 1)
- ✅ 自动提取和对比性能数据
- ✅ 所有测试使用相同配置（唯一真相源）

### 方式 2: 手动测试（仅用于调试）⚠️

```bash
# 1. 检查环境变量
echo "METAL_FUSION=$METAL_FUSION (应该是 1)"
echo "FUSED_QKV=$FUSED_QKV (应该是 1)"
echo "USE_MPS_GRAPH=$USE_MPS_GRAPH (应该是 0)"
echo "THUNDER_LMCACHE=$THUNDER_LMCACHE (应该是 0)"
echo "LLAMA_PAGED_ATTENTION=$LLAMA_PAGED_ATTENTION (应该是 0)"

# 2. 设置环境变量（如果未设置）
export METAL_FUSION=1
export FUSED_QKV=1
export USE_MPS_GRAPH=0
export THUNDER_LMCACHE=0
export LLAMA_PAGED_ATTENTION=0

# 3. 执行测试（注意：-t 4 不是 -t 10！）
./build/bin/llama-bench \
    -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q4_K_M.gguf \
    -t 4 -ngl 99 -r 5 -p 512 -n 128 -fa 1

# 4. 验证结果
# 应该看到 TG ≈ 79.74 tok/s (Q4_K_M, 正确参数)
```

---

## ⚠️ 常见错误

| 错误 | 表现 | 根因 | 解决 |
|------|------|------|------|
| **虚假回退 11%** ⚠️ | 70.39 vs 79.12 | **错误参数: -t 10 而非 -t 4** | **使用 scripts/benchmark-standard.sh** |
| 虚假回退 8.4% | 72.51 vs 79.12 | METAL_FUSION 未设置 | 设置 METAL_FUSION=1 |
| 虚假回退 3.4% | 76.88 vs 79.12 | N_R0_Q4_K 错误设置为 8 | 恢复 N_R0_Q4_K=2 |
| 输出不一致 | 不同运行结果不同 | FUSED_QKV 未固定 | 固定 FUSED_QKV=1 |
| 测试参数不一致 | 每次结果不同 | 重复创建新的测试脚本 | **只使用 scripts/benchmark-standard.sh** |

**🔥 最常见错误**: 使用 `-t 10` (默认值) 而非 `-t 4` (M4 Pro 优化值)，导致 11% 虚假性能回退！

---

## 📊 历史基准记录

| 日期 | 模型 | TG (tok/s) | PP (tok/s) | 配置 | 备注 |
|------|------|-----------|-----------|------|------|
| 2026-03-15 | Q4_K_M | **79.79 ± 0.14** | 784.62 ± 3.93 | 向量化 SWIGLU, -t 4 -fa 1 | A4 优化 (+0.06%) |
| 2026-03-15 | Q4_K_M | **79.74 ± 0.10** | 785.88 ± 4.98 | 标量 SWIGLU, -t 4 -fa 1 | A4 对比基线 |
| 2026-03-15 | Q4_K_M | **79.97 ± 0.41** | 779.93 ± 8.66 | 正确参数验证, -t 4 -fa 1 | 参数修正后 |
| 2026-03-15 | Q4_K_M | 79.12 ± 0.20 | 787.50 ± 6.14 | build 8389+MoE fusion | 历史记录 |
| 2026-03-14 | Q5_K_M | 65.25 ± 0.16 | - | METAL_FUSION=1, N_R0=8 | Q5_K 基准 |

**重要发现**: `-t 4` (M4 Pro 优化) 比 `-t 10` (默认) 快 **11%** (79.97 vs 70.39 tok/s)

---

## 🎯 标准化测试脚本

**位置**: `/Users/lisihao/ThunderLLAMA/scripts/benchmark-standard.sh`

**所有性能测试必须使用此脚本**，不再临时编写新的测试脚本。

**设计目标**:
- 唯一真相源: 所有参数集中在脚本顶部配置区
- 防止人为错误: 自动设置所有必需的环境变量
- 结果可比性: 确保每次测试使用相同参数

---

*最后更新: 2026-03-15*
*标准化脚本创建: 2026-03-15*
*作用: 防止系统性遗漏关键配置，避免测试参数不一致*
