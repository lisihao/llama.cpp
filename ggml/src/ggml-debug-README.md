# GGML Debug Logging Utilities

通用调试工具库，用于 ThunderLLAMA 开发和问题诊断。

## 特性

- ✅ 环境变量控制（`GGML_DEBUG=1`）
- ✅ 时间戳（精确到微秒）
- ✅ 文件名和行号
- ✅ 自动 flush（不丢失日志）
- ✅ 多种日志级别和类型
- ✅ 性能计时工具
- ✅ 零运行时开销（未启用时）

## 快速开始

### 1. 引入头文件

```cpp
#include "ggml-debug.h"
```

### 2. 使用调试宏

```cpp
DEBUG_LOG("Hello from %s", __func__);
DEBUG_LOG_INT("batch_size", 32);
DEBUG_LOG_PTR("buffer", my_buffer);
```

### 3. 启用调试输出

```bash
export GGML_DEBUG=1
./build/bin/llama-server
```

## 宏列表

### 基础日志

| 宏 | 用途 | 示例 |
|----|------|------|
| `DEBUG_LOG(fmt, ...)` | 通用日志（带时间戳、位置） | `DEBUG_LOG("Processing %d items", count)` |
| `DEBUG_LOG_TAG(tag, fmt, ...)` | 带标签的日志 | `DEBUG_LOG_TAG("METAL", "Kernel launched")` |
| `DEBUG_LOG_SIMPLE(fmt, ...)` | 简单日志（无位置信息） | `DEBUG_LOG_SIMPLE("=== Section Start ===")` |

### 类型化日志

| 宏 | 用途 | 示例 |
|----|------|------|
| `DEBUG_LOG_INT(name, val)` | 打印整数 | `DEBUG_LOG_INT("seq_len", 2048)` |
| `DEBUG_LOG_FLOAT(name, val)` | 打印浮点数 | `DEBUG_LOG_FLOAT("temperature", 0.7)` |
| `DEBUG_LOG_PTR(name, ptr)` | 打印指针地址 | `DEBUG_LOG_PTR("kv_cache", cache_ptr)` |
| `DEBUG_LOG_TENSOR(name, tensor)` | 打印 tensor 信息 | `DEBUG_LOG_TENSOR("input", input_tensor)` |
| `DEBUG_LOG_BUFFER(name, buf, size, max)` | 打印 buffer 内容（十六进制） | `DEBUG_LOG_BUFFER("data", buf, 64, 16)` |

### 条件日志

| 宏 | 用途 | 示例 |
|----|------|------|
| `DEBUG_LOG_IF(cond, fmt, ...)` | 仅当条件为真时打印 | `DEBUG_LOG_IF(error_count > 0, "Errors: %d", error_count)` |

### 性能计时

| 宏 | 用途 | 示例 |
|----|------|------|
| `DEBUG_TIMER_START(name)` | 开始计时 | `DEBUG_TIMER_START(kernel_exec)` |
| `DEBUG_TIMER_END(name)` | 结束计时并打印耗时（毫秒） | `DEBUG_TIMER_END(kernel_exec)` |

### 断言

| 宏 | 用途 | 示例 |
|----|------|------|
| `DEBUG_ASSERT(cond, fmt, ...)` | 条件不满足时中止并打印消息 | `DEBUG_ASSERT(ptr != NULL, "Null pointer!")` |

## 使用示例

### 示例 1: Metal Kernel 调试

```cpp
#include "ggml-debug.h"

void ggml_metal_kernel_launch(...) {
    DEBUG_LOG_TAG("METAL", "Launching kernel: %s", kernel_name);

    DEBUG_LOG_INT("batch_size", batch_size);
    DEBUG_LOG_INT("seq_len", seq_len);
    DEBUG_LOG_PTR("input_buffer", input_buffer);

    DEBUG_TIMER_START(kernel_exec);
    // ... kernel execution
    DEBUG_TIMER_END(kernel_exec);

    DEBUG_LOG_TAG("METAL", "Kernel complete");
}
```

### 示例 2: Paged Attention 调试

```cpp
#include "ggml-debug.h"

void setup_paged_attention(...) {
    DEBUG_LOG("=== Paged Attention Setup ===");

    DEBUG_LOG_INT("kv_size", kv_size);
    DEBUG_LOG_INT("block_size", 16);
    DEBUG_LOG_INT("max_blocks", 256);

    if (block_table) {
        DEBUG_LOG_BUFFER("block_table", block_table, 10 * sizeof(int32_t), 40);
    }

    DEBUG_ASSERT(kv_size % block_size == 0,
                 "KV size (%d) must be divisible by block size (%d)",
                 kv_size, block_size);
}
```

### 示例 3: KV Cache 状态追踪

```cpp
#include "ggml-debug.h"

void update_kv_cache(...) {
    DEBUG_LOG_TAG("CACHE", "Updating KV cache");

    DEBUG_TIMER_START(cache_update);

    for (int i = 0; i < n_tokens; i++) {
        int physical_idx = block_table[i / block_size] * block_size + (i % block_size);

        DEBUG_LOG_IF(i < 5, "Token %d: logical=%d, physical=%d",
                     i, i, physical_idx);
    }

    DEBUG_TIMER_END(cache_update);
}
```

### 示例 4: 错误诊断

```cpp
#include "ggml-debug.h"

void diagnose_output() {
    DEBUG_LOG("=== Output Diagnosis ===");

    DEBUG_LOG_TENSOR("output", output_tensor);

    // Check for NaN
    float* data = (float*)output_tensor->data;
    int nan_count = 0;
    for (int i = 0; i < 100; i++) {
        if (isnan(data[i])) nan_count++;
    }

    DEBUG_LOG_IF(nan_count > 0, "⚠️  Found %d NaN values!", nan_count);

    // Print first few values
    DEBUG_LOG_SIMPLE("First 10 output values:");
    for (int i = 0; i < 10; i++) {
        DEBUG_LOG_SIMPLE("  [%d] = %.6f", i, data[i]);
    }
}
```

## 编译示例程序

```bash
cd /Users/lisihao/ThunderLLAMA/ggml/src

# 编译示例
g++ -o debug-example ggml-debug-example.cpp

# 运行（调试关闭）
./debug-example

# 运行（调试开启）
GGML_DEBUG=1 ./debug-example
```

## 集成到现有代码

### 1. 在 Metal ops 中使用

```cpp
// ggml-metal-ops.cpp
#include "ggml-debug.h"

int ggml_metal_op_flash_attn_ext(...) {
    DEBUG_LOG_TAG("METAL_FATTN", "Entry point");
    DEBUG_LOG_INT("idx", idx);

    // ... existing code ...

    DEBUG_LOG_IF(has_paged, "Paged Attention enabled");
}
```

### 2. 在 Graph 构建中使用

```cpp
// llama-graph.cpp
#include "ggml-debug.h"

void build_graph(...) {
    DEBUG_LOG_TAG("GRAPH", "Building graph for %s", model_name);

    DEBUG_TIMER_START(graph_build);
    // ... build graph ...
    DEBUG_TIMER_END(graph_build);
}
```

### 3. 在 KV Cache 中使用

```cpp
// llama-kv-cache.cpp
#include "ggml-debug.h"

void allocate_kv_cache(...) {
    DEBUG_LOG_TAG("KV_CACHE", "Allocating cache");
    DEBUG_LOG_INT("capacity", capacity);

    DEBUG_ASSERT(capacity > 0, "Invalid capacity: %d", capacity);
}
```

## 性能影响

- **未启用时**（`GGML_DEBUG=0` 或未设置）：
  - 零运行时开销（宏展开为空）
  - 编译器会优化掉所有调试代码

- **启用时**（`GGML_DEBUG=1`）：
  - fprintf 系统调用开销
  - 建议仅在开发/调试时启用

## 注意事项

1. **生产环境**：不要在生产环境设置 `GGML_DEBUG=1`
2. **敏感数据**：避免打印敏感信息（密钥、用户数据）
3. **性能**：调试日志会影响性能，仅在需要时启用
4. **线程安全**：fprintf 本身是线程安全的，但多线程输出可能交错

## 常见场景

### 场景 1: 调试乱码输出

```cpp
DEBUG_LOG_TAG("OUTPUT", "Checking output sanity");
DEBUG_LOG_BUFFER("raw_output", output_bytes, 64, 64);
DEBUG_LOG_TENSOR("output_tensor", output);
```

### 场景 2: 追踪内存分配

```cpp
DEBUG_TIMER_START(allocation);
void* buffer = malloc(size);
DEBUG_TIMER_END(allocation);

DEBUG_LOG_PTR("allocated_buffer", buffer);
DEBUG_LOG_INT("size", size);
```

### 场景 3: 验证算法正确性

```cpp
DEBUG_LOG("=== Verification ===");

for (int i = 0; i < n; i++) {
    int expected = compute_expected(i);
    int actual = array[i];

    DEBUG_LOG_IF(expected != actual,
                 "Mismatch at %d: expected=%d, actual=%d",
                 i, expected, actual);
}
```

## 与现有日志系统的关系

- **LLAMA_LOG_***：llama.cpp 的日志系统（用于用户可见的信息）
- **DEBUG_LOG_***：开发调试系统（仅开发者使用，需环境变量启用）

建议：
- 用户相关信息 → 使用 LLAMA_LOG_INFO/WARN/ERROR
- 开发调试信息 → 使用 DEBUG_LOG_*

## 扩展

如果需要更高级的功能，可以扩展 `ggml-debug.h`：

- 添加日志级别过滤（ERROR, WARN, INFO, DEBUG）
- 添加日志文件输出（不只是 stderr）
- 添加日志格式化选项（JSON, CSV）
- 添加远程日志传输（网络）

---

**创建时间**: 2026-03-16
**作者**: ThunderLLAMA Team
**版本**: 1.0
