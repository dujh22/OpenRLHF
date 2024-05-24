set -x 

read -r -d '' training_commands <<EOF
../train_rm.py \
     --save_path ./ckpt/8b_llama3 \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain /workspace/dujh22/models/llama-3-8B \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --dataset /workspace/dujh22/openrlhf/Anthropichh_hh_rlhf \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb True
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --include="localhost:2,3" $training_commands
fi
