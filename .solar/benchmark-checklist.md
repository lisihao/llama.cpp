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

### 方式 1: 使用标准化脚本（推荐）

```bash
./bench-standard.sh  # 自动检查 + 自动验证
```

### 方式 2: 手动测试（需严格遵循检查清单）

```bash
# 1. 检查环境变量
echo "METAL_FUSION=$METAL_FUSION (应该是 1)"
echo "FUSED_QKV=$FUSED_QKV (应该是 1)"

# 2. 设置环境变量（如果未设置）
export METAL_FUSION=1
export FUSED_QKV=1
export USE_MPS_GRAPH=0

# 3. 执行测试
METAL_FUSION=1 FUSED_QKV=1 USE_MPS_GRAPH=0 \
./build/bin/llama-bench \
    -m ~/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-128K-Q4_K_M.gguf \
    -t 4 -ngl 99 -r 5 -p 0 -n 128 -fa 1

# 4. 验证结果
# 应该看到 TG ≈ 79.12 tok/s (Q4_K)
```

---

## ⚠️ 常见错误

| 错误 | 表现 | 根因 | 解决 |
|------|------|------|------|
| 性能回退 8.4% | 72.51 vs 79.12 | METAL_FUSION 未设置 | 设置 METAL_FUSION=1 |
| 性能回退 3.4% | 76.88 vs 79.12 | N_R0_Q4_K 错误设置为 8 | 恢复 N_R0_Q4_K=2 |
| 输出不一致 | 不同运行结果不同 | FUSED_QKV 未固定 | 固定 FUSED_QKV=1 |

---

## 📊 历史基准记录

| 日期 | 模型 | TG (tok/s) | 配置 | 备注 |
|------|------|-----------|------|------|
| 2026-03-15 | Q4_K_M | 79.81 ± 0.15 | METAL_FUSION=1, FUSED_QKV=1 | 性能回退修复后 |
| 2026-03-15 | Q4_K_M | 79.12 ± 0.20 | build 8389+MoE fusion | 记录基准 |
| 2026-03-14 | Q5_K_M | 65.25 ± 0.16 | METAL_FUSION=1, N_R0=8 | Q5_K 基准 |

---

*最后更新: 2026-03-15*
*作用: 防止系统性遗漏关键配置*
