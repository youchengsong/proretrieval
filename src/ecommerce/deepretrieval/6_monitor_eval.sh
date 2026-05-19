#!/bin/bash
# Monitor for new DeepRetrieval checkpoints and auto-evaluate on GPU 0.
# Usage: nohup bash 6_monitor_eval.sh > eval_monitor.log 2>&1 &

set -e
cd /home/sunzhen/DeepRetrieval_ecommerce

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=~/miniconda3/envs/mlx/bin/python
EVAL_SCRIPT=/home/sunzhen/DeepRetrieval_ecommerce/5_eval_checkpoint.py
CKPT_BASE=/home/sunzhen/DeepRetrieval_ecommerce/checkpoints
EVAL_DIR=/home/sunzhen/DeepRetrieval_ecommerce/eval_results
DONE_FILE=${EVAL_DIR}/.evaluated_steps

mkdir -p $EVAL_DIR
touch $DONE_FILE

echo "$(date) - Starting checkpoint monitor..."
echo "Checkpoint dir: $CKPT_BASE"
echo "Eval results:   $EVAL_DIR"
echo "Using GPU:      $CUDA_VISIBLE_DEVICES"
echo ""

while true; do
    # Find all global_step_* directories
    for ckpt_dir in $(find $CKPT_BASE -maxdepth 3 -name "global_step_*" -type d 2>/dev/null | sort -t_ -k3 -n); do
        step_name=$(basename $ckpt_dir)

        # Skip if already evaluated
        if grep -q "^${step_name}$" $DONE_FILE 2>/dev/null; then
            continue
        fi

        # Check if checkpoint has config.json (fully saved)
        model_dir=""
        if [ -f "${ckpt_dir}/actor/config.json" ]; then
            model_dir="${ckpt_dir}/actor"
        elif [ -f "${ckpt_dir}/actor/huggingface/config.json" ]; then
            model_dir="${ckpt_dir}/actor/huggingface"
        elif [ -f "${ckpt_dir}/config.json" ]; then
            model_dir="${ckpt_dir}"
        fi

        if [ -z "$model_dir" ]; then
            echo "$(date) - ${step_name}: no config.json found yet, skipping..."
            continue
        fi

        # Check if model files exist (not just config)
        n_safetensors=$(find "$model_dir" -name "*.safetensors" 2>/dev/null | wc -l)
        n_bin=$(find "$model_dir" -name "*.bin" 2>/dev/null | wc -l)
        if [ $((n_safetensors + n_bin)) -eq 0 ]; then
            echo "$(date) - ${step_name}: no model weights found yet, skipping..."
            continue
        fi

        echo ""
        echo "$(date) - Found new checkpoint: ${step_name}"
        echo "  Model dir: ${model_dir}"
        echo "  Running evaluation..."

        # Run eval (try vLLM first, fallback to transformers)
        if $PYTHON $EVAL_SCRIPT --checkpoint "$ckpt_dir" --gpu-mem 0.22 2>&1; then
            echo "${step_name}" >> $DONE_FILE
            echo "$(date) - ${step_name}: evaluation complete!"
        else
            echo "$(date) - ${step_name}: vLLM failed, retrying with transformers..."
            if $PYTHON $EVAL_SCRIPT --checkpoint "$ckpt_dir" --use-transformers 2>&1; then
                echo "${step_name}" >> $DONE_FILE
                echo "$(date) - ${step_name}: evaluation complete (transformers)!"
            else
                echo "$(date) - ${step_name}: EVALUATION FAILED"
                echo "${step_name}" >> $DONE_FILE  # Don't retry failed evals
            fi
        fi

        # Print running summary
        echo ""
        echo "=== Running Summary ==="
        if [ -f ${EVAL_DIR}/summary.txt ]; then
            cat ${EVAL_DIR}/summary.txt
        fi
        echo "======================="
        echo ""
    done

    # Check if training is still running
    n_train=$(ps aux | grep "verl.trainer" | grep -v grep | wc -l)
    if [ $n_train -eq 0 ]; then
        # Check one more time for final checkpoint
        sleep 30
        n_new=$(find $CKPT_BASE -maxdepth 3 -name "global_step_*" -type d 2>/dev/null | while read d; do
            s=$(basename $d)
            grep -q "^${s}$" $DONE_FILE 2>/dev/null || echo "new"
        done | wc -l)
        if [ $n_new -eq 0 ]; then
            echo "$(date) - Training finished. No more checkpoints to evaluate."
            echo ""
            echo "=== Final Summary ==="
            [ -f ${EVAL_DIR}/summary.txt ] && cat ${EVAL_DIR}/summary.txt
            echo "====================="
            break
        fi
    fi

    sleep 300  # Check every 5 minutes
done
