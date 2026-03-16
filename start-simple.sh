#!/bin/bash
# 简化版启动脚本 - 直接从 thunderllama.conf 读取配置并启动

set -e

cd "$(dirname "$0")"

# 读取配置
eval "$(grep -v '^#' thunderllama.conf | grep -v '^$' | sed 's/#.*//' | sed 's/[[:space:]]*$//')"

# 展开模型路径
MODEL_PATH_EXPANDED="${MODEL_PATH/#\~/$HOME}"

echo "=== ThunderLLAMA 启动 ==="
echo "模型: $MODEL_PATH_EXPANDED"
echo "端口: $SERVER_PORT"

# 停止现有服务
if lsof -ti:$SERVER_PORT > /dev/null 2>&1; then
    echo "停止现有服务..."
    kill $(lsof -ti:$SERVER_PORT) || true
    sleep 2
fi

# 设置环境变量
export THUNDER_LMCACHE="$THUNDER_LMCACHE"
export THUNDER_LMCACHE_DISK_PATH="$THUNDER_LMCACHE_DISK_PATH"
export THUNDER_PREFIX_MATCHING="$THUNDER_PREFIX_MATCHING"
export LLAMA_PAGED_ATTENTION="$LLAMA_PAGED_ATTENTION"
export THUNDERLLAMA_CHUNK_PREFILL="$THUNDERLLAMA_CHUNK_PREFILL"
export GGML_METAL_MAX_BUFFER_SIZE="$METAL_MAX_BUFFER_SIZE"

# Metal Fusion (如果禁用)
if [ "$METAL_FUSION" = "0" ]; then
    export GGML_METAL_FUSION_DISABLE=1
fi

# 构建命令
CMD="./build/bin/llama-server"
CMD="$CMD -m '$MODEL_PATH_EXPANDED'"
CMD="$CMD -c $CONTEXT_SIZE"
CMD="$CMD -ngl $GPU_LAYERS"
CMD="$CMD -t $CPU_THREADS"
CMD="$CMD -tb $CPU_THREADS_BATCH"
CMD="$CMD --parallel $PARALLEL_SLOTS"
CMD="$CMD -b $BATCH_SIZE"
CMD="$CMD -ub $UBATCH_SIZE"
CMD="$CMD --cache-reuse $CACHE_REUSE"
CMD="$CMD --cache-ram $CACHE_RAM"
CMD="$CMD --port $SERVER_PORT"

if [ "$FLASH_ATTENTION" = "on" ]; then
    CMD="$CMD -fa"
fi

if [ "$KV_UNIFIED" = "1" ]; then
    CMD="$CMD --kv-unified"
fi

if [ "$CONT_BATCHING" = "1" ]; then
    CMD="$CMD --cont-batching"
fi

if [ -n "$PRIO_BATCH" ]; then
    CMD="$CMD --prio-batch $PRIO_BATCH"
fi

if [ "$GRAPH_REUSE" = "1" ]; then
    CMD="$CMD --graph-reuse"
fi

# 日志
CMD="$CMD > $LOG_FILE 2>&1 &"

echo ""
echo "启动命令:"
echo "$CMD"
echo ""

# 启动
eval $CMD
PID=$!

echo "服务器已启动 (PID: $PID)"
echo "日志: $LOG_FILE"

# 等待就绪
echo "等待服务器就绪..."
sleep 10

for i in {1..30}; do
    if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
        echo "✅ 服务器就绪！"

        # 配置 KV Cache
        if [ -n "$KV_CACHE_LEVEL" ]; then
            echo "配置 KV Cache: $KV_CACHE_LEVEL"
            curl -s http://localhost:$SERVER_PORT/thunder/kv-strategy \
                -X POST \
                -H "Content-Type: application/json" \
                -d "{\"name\":\"$KV_CACHE_STRATEGY\",\"params\":{\"level\":\"$KV_CACHE_LEVEL\"},\"version\":1}" | jq .
        fi

        echo ""
        echo "=========================================="
        echo "ThunderLLAMA 已启动成功！"
        echo "=========================================="
        echo "PID: $PID"
        echo "端口: $SERVER_PORT"
        echo "日志: tail -f $LOG_FILE"
        echo "停止: kill $PID"
        echo "=========================================="

        exit 0
    fi
    sleep 1
done

echo "❌ 服务器启动失败"
echo "查看日志: tail -100 $LOG_FILE"
exit 1
