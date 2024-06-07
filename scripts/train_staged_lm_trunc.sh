#!/bin/bash
ulimit -m 12000000
source $HOME/ma/alti/venv_alti/bin/activate
ORIG=$HOME/ma/data/sentp/iwslt14.sep.tokenized.de-en
TEXT=./data/iwslt14.sep.tokenized.de-en

# remove data from source file
mkdir -p data
cp -r $ORIG data
cd $TEXT

cp $HOME/ma/scripts/clean_lm_trunc.py .
python clean_lm_trunc.py
cd ../../


fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.sep.tokenized.de-en \
    --workers 20

CUDA_VISIBLE_DEVICES=7 fairseq-train \
    data-bin/iwslt14.sep.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --no-epoch-checkpoints \
    --save-interval-updates 100 --max-update 1000 

CUDA_VISIBLE_DEVICES=7 fairseq-train \
    data-bin/iwslt14.sep.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --no-epoch-checkpoints \
    --save-interval-updates 500 --max-update 10000

CUDA_VISIBLE_DEVICES=7 fairseq-train \
    data-bin/iwslt14.sep.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --no-epoch-checkpoints \
    --save-interval-updates 1000 --max-update 100000

#CUDA_VISIBLE_DEVICES=0 fairseq-train \
#    data-bin/iwslt14.sep.tokenized.de-en \
#    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 4096 \
#    --no-epoch-checkpoints --disable-validation --no-last-checkpoints \
#    --save-interval-updates 50000 --max-update 1000000
