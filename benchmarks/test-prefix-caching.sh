#!/bin/bash
#
# Prefix Caching Test for ThunderLLAMA
# Tests block reuse in multi-turn conversation scenarios
#

set -e

# Configuration
MODEL="${1:-$HOME/models/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf}"
BIN_DIR="./build/bin"
CTX=4096
BLOCK_SIZE=16

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     ThunderLLAMA Prefix Caching Test                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL"

if [ ! -f "$MODEL" ]; then
    echo -e "${RED}Error: Model not found: $MODEL${NC}"
    echo "Usage: $0 <path-to-model.gguf>"
    exit 1
fi

MODEL_SIZE=$(du -h "$MODEL" | cut -f1)
echo "Size: $MODEL_SIZE"
echo ""

# Test prompt - System prompt + multi-turn conversation
# The system prompt is the "prefix" that should be cached across turns

SYSTEM_PROMPT="You are a helpful AI assistant specialized in technical explanations. You provide clear, accurate, and detailed answers about programming, software engineering, and computer science topics. Always structure your responses with examples when possible."

# Turn 1 - First request (cold cache)
TURN1_USER="Explain what is paged attention in LLM inference."

# Turn 2 - Second request (should reuse prefix)
TURN2_USER="What are the benefits of paged attention over contiguous KV cache?"

# Turn 3 - Third request (should reuse prefix)
TURN3_USER="How does copy-on-write work in prefix caching?"

# Full prompts
PROMPT1="$SYSTEM_PROMPT

User: $TURN1_USER
Assistant:"

PROMPT2="$SYSTEM_PROMPT

User: $TURN1_USER
Assistant: [Previous response about paged attention...]

User: $TURN2_USER
Assistant:"

PROMPT3="$SYSTEM_PROMPT

User: $TURN1_USER
Assistant: [Previous response...]

User: $TURN2_USER
Assistant: [Previous response about benefits...]

User: $TURN3_USER
Assistant:"

echo "════════════════════════════════════════════════════════════════════"
echo "Test 1: Baseline - No Prefix Caching"
echo "════════════════════════════════════════════════════════════════════"

# Run Turn 1 without prefix caching (baseline)
echo ""
echo -e "${BLUE}[Turn 1] First request (cold start)${NC}"
TIME1_START=$(date +%s%N)
$BIN_DIR/llama-cli \
    -m "$MODEL" \
    -p "$PROMPT1" \
    -c $CTX \
    -n 32 \
    -fa 1 \
    -ngl 99 \
    --temp 0.0 \
    2>&1 | grep -E "eval time|tokens per second" || true
TIME1_END=$(date +%s%N)
TIME1_MS=$(( (TIME1_END - TIME1_START) / 1000000 ))
echo -e "Total time: ${YELLOW}${TIME1_MS}ms${NC}"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Test 2: With Prefix Caching"
echo "════════════════════════════════════════════════════════════════════"

# Enable prefix caching
export LLAMA_PAGED_ATTENTION=1
export LLAMA_PREFIX_CACHING=1

echo ""
echo -e "${BLUE}[Turn 1] First request (building cache)${NC}"
TIME2_START=$(date +%s%N)
$BIN_DIR/llama-cli \
    -m "$MODEL" \
    -p "$PROMPT1" \
    -c $CTX \
    -n 32 \
    -fa 1 \
    -ngl 99 \
    --temp 0.0 \
    2>&1 | grep -E "eval time|tokens per second|prefix|cached" || true
TIME2_END=$(date +%s%N)
TIME2_MS=$(( (TIME2_END - TIME2_START) / 1000000 ))
echo -e "Total time: ${YELLOW}${TIME2_MS}ms${NC}"

echo ""
echo -e "${BLUE}[Turn 2] Second request (should reuse prefix)${NC}"
TIME3_START=$(date +%s%N)
$BIN_DIR/llama-cli \
    -m "$MODEL" \
    -p "$PROMPT2" \
    -c $CTX \
    -n 32 \
    -fa 1 \
    -ngl 99 \
    --temp 0.0 \
    2>&1 | grep -E "eval time|tokens per second|prefix|cached|reused" || true
TIME3_END=$(date +%s%N)
TIME3_MS=$(( (TIME3_END - TIME3_START) / 1000000 ))
echo -e "Total time: ${YELLOW}${TIME3_MS}ms${NC}"

echo ""
echo -e "${BLUE}[Turn 3] Third request (longer context, reuse prefix)${NC}"
TIME4_START=$(date +%s%N)
$BIN_DIR/llama-cli \
    -m "$MODEL" \
    -p "$PROMPT3" \
    -c $CTX \
    -n 32 \
    -fa 1 \
    -ngl 99 \
    --temp 0.0 \
    2>&1 | grep -E "eval time|tokens per second|prefix|cached|reused" || true
TIME4_END=$(date +%s%N)
TIME4_MS=$(( (TIME4_END - TIME4_START) / 1000000 ))
echo -e "Total time: ${YELLOW}${TIME4_MS}ms${NC}"

# Cleanup
unset LLAMA_PREFIX_CACHING

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Summary"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Note: Current implementation requires integration into the CLI to"
echo "      show prefix reuse statistics. The infrastructure is in place"
echo "      but needs hooking into the decoding loop."
echo ""
echo "Prefix caching API is available:"
echo "  - llama_block_pool::set_prefix_caching(bool)"
echo "  - llama_block_pool::find_block_by_hash(hash)"
echo "  - llama_block_pool::share_block(block_id, seq_id)"
echo "  - llama_kv_cache::try_reuse_prefix(seq_id, tokens, n_tokens)"
echo ""
echo "To fully utilize prefix caching, integrate it into your application:"
echo "  1. Enable: kv_cache->set_prefix_caching(true)"
echo "  2. Before prefill: try_reuse_prefix(seq_id, tokens, n_tokens)"
echo "  3. After prefill: register_prefix_blocks(seq_id, tokens, n_tokens)"
echo ""
