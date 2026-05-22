#!/bin/bash
# Master orchestration: SQL-R1 first, then Search-R1
# Both run sequentially on 2 GPUs
set -x

LOG_DIR=/root/autodl-tmp/logs
mkdir -p $LOG_DIR
MASTER_LOG=$LOG_DIR/master_progress.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $MASTER_LOG
}

log "============================================"
log "  MASTER PIPELINE STARTING"
log "  SQL-R1 (ecommerce+email) -> Search-R1 (ecommerce+email)"
log "============================================"

# Set HuggingFace mirror for China
export HF_ENDPOINT=https://hf-mirror.com

# Install trl if missing
pip install trl 2>/dev/null
pip install rank_bm25 2>/dev/null

# ── Phase 1: SQL-R1 ──
log "PHASE 1: SQL-R1 Pipeline"
bash /root/autodl-tmp/scripts/run_sql_r1_pipeline.sh 2>&1
SQL_EXIT=$?
if [ $SQL_EXIT -ne 0 ]; then
    log "WARNING: SQL-R1 pipeline exited with code $SQL_EXIT. Continuing to Search-R1..."
fi

# ── Phase 2: Search-R1 ──
log "PHASE 2: Search-R1 Pipeline"
bash /root/autodl-tmp/scripts/run_search_r1_pipeline.sh 2>&1
SEARCH_EXIT=$?
if [ $SEARCH_EXIT -ne 0 ]; then
    log "WARNING: Search-R1 pipeline exited with code $SEARCH_EXIT."
fi

log "============================================"
log "  MASTER PIPELINE COMPLETE"
log "  SQL-R1 exit: $SQL_EXIT"
log "  Search-R1 exit: $SEARCH_EXIT"
log "============================================"

# Print summary
log "Checkpoints:"
ls -la /root/autodl-tmp/sql_r1/checkpoints/*/final/ 2>/dev/null | tee -a $MASTER_LOG
ls -la /root/autodl-tmp/search_r1/checkpoints/*/final/ 2>/dev/null | tee -a $MASTER_LOG

log "Logs:"
ls -la $LOG_DIR/*.log | tee -a $MASTER_LOG
