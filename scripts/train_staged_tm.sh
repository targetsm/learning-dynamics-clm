#!/bin/bash
source $HOME/ma/alti/venv_alti/bin/activate
TEXT=$HOME/ma/data/sentp/iwslt14.sep.tokenized.de-en

fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.sep.tokenized.de-en \
    --workers 20

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.sep.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --maximize-best-checkpoint-metric --no-epoch-checkpoints \
    --save-interval-updates 100 --max-update 1000 

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.sep.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --maximize-best-checkpoint-metric --no-epoch-checkpoints \
    --save-interval-updates 500 --max-update 10000

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.sep.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --maximize-best-checkpoint-metric --no-epoch-checkpoints \
    --save-interval-updates 1000 --max-update 100000

#CUDA_VISIBLE_DEVICES=0 fairseq-train \
#    data-bin/iwslt14.tokenized.de-en \
#    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 4096 \
#    --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
#    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#    --save-interval-updates 50000 --max-update 1000000
#
#fairseq-generate data-bin/iwslt14.sep.tokenized.de-en \
#    --path checkpoints/checkpoint_best.pt \
#    --batch-size 128 --beam 5 --remove-bpe
