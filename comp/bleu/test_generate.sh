#!/bin/bash
ulimit -m 12000000
source $HOME/ma/alti/venv_alti/bin/activate
cd $HOME/ma/models/tl/wmt22frde
mkdir -p evaluation_generate

for filename in checkpoints/analysis/*; do
    echo $filename
    CUDA_VISIBLE_DEVICES=3 fairseq-generate data-bin/iwslt14.sep.tokenized.de-en \
        --path $filename \
        --batch-size 256 --beam 5\
        --results-path evaluation_generate/$(basename $filename .pt)
done
