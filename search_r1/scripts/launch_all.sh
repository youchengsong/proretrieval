#!/bin/bash
set -e

LOG_DIR=/root/autodl-tmp/logs
mkdir -p $LOG_DIR

echo "[$(date)] Starting retrieval server..." | tee -a $LOG_DIR/search_rl_v3_progress.log

# Kill any existing sessions
screen -S retriever -X quit 2>/dev/null || true
screen -S rl_train -X quit 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2

# Start retrieval server in screen
screen -dmS retriever bash -c "
cd /root/autodl-tmp/Search-R1
python3 search_r1/search/retrieval_server.py \
    --retriever_name bm25 \
    --index_path /root/autodl-tmp/data/ecommerce/bm25_index \
    --corpus_path /root/autodl-tmp/data/ecommerce/bm25_corpus/corpus.jsonl \
    --topk 3 \
    2>&1 | tee $LOG_DIR/retrieval_server.log
"

echo "[$(date)] Waiting for retrieval server to start..." | tee -a $LOG_DIR/search_rl_v3_progress.log

# Wait for server to be ready
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8000/retrieve -X POST -H "Content-Type: application/json" -d '{"queries":["red shoes"],"topk":3}' > /dev/null 2>&1; then
        echo "[$(date)] Retrieval server is UP" | tee -a $LOG_DIR/search_rl_v3_progress.log
        break
    fi
    echo "  waiting... ($i/30)"
    sleep 2
done

# Test retrieval
echo "[$(date)] Testing retrieval..." | tee -a $LOG_DIR/search_rl_v3_progress.log
curl -s http://127.0.0.1:8000/retrieve -X POST -H "Content-Type: application/json" \
    -d '{"queries":["red shoes for women"],"topk":3}' 2>&1 | python3 -c "
import sys, json
resp = json.load(sys.stdin)
print(f'Retrieved {len(resp.get(\"passages\", resp.get(\"results\", [])))} passages')
print(json.dumps(resp, indent=2)[:500])
" 2>&1 || echo "Retrieval test response format check needed"

# Start RL training in screen
echo "[$(date)] Starting RL training..." | tee -a $LOG_DIR/search_rl_v3_progress.log
screen -dmS rl_train bash -c "
    bash /root/autodl-tmp/scripts/run_search_rl_v3.sh 2>&1
"

echo "[$(date)] RL training launched in screen 'rl_train'" | tee -a $LOG_DIR/search_rl_v3_progress.log
echo "Check with: screen -r rl_train"
echo "Retrieval server in: screen -r retriever"
