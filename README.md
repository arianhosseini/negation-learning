
# Negation-Learning

# train BERETNOT

```
python -m torch.distributed.launch --nproc_per_node=1 \ examples/lm_training/finetune_on_pregenerated_negation_distributed.py\
    --pregenerated_neg_data path/to/wiki20k_negated_withref_UL_pregenerated/ \    
    --pregenerated_pos_data path/to/wiki20k_positive_withref_LL_pregenerated/ \  
    --validation_neg_data path/to/validation/neg/ \
    --validation_pos_data path/to/validation/pos/ \
    --pregenerated_data path/to/lm_training_wiki20k/ \
    --output_dir path/to/output \
    --bert_model bert-base-cased \
    --fp16 \
    --exp_group neg_sb_gamma0.0_lr1e_5_e5_wiki20k \
    --learning_rate 1e-5 \
    --port_idx 0 \
    --kr_freq 0.0 \
    --train_batch_size 32 \
    --mlm_freq 0.0 \
    --epoch 5 \
    --gamma 0.0 \
    --seed $seed \
    --method neg_samebatch
```
