export HANS_DIR=~/hans
export MODEL_TYPE=bert
export MODEL_PATH=/workdrive/test/glue_results/MNLI/

python test_hans.py \
        --task_name hans \
        --model_type $MODEL_TYPE \
        --do_eval \
        --data_dir $HANS_DIR \
        --model_name_or_path $MODEL_PATH \
        --max_seq_length 128 \
        --output_dir $MODEL_PATH \
