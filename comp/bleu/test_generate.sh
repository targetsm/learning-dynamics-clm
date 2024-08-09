#!/bin/bash
ulimit -m 12000000
source $HOME/ma/alti/venv_alti/bin/activate
cd $HOME/ma/models/tl/wmt22frde_subset/wmt_big/
mkdir -p evaluation_generate

for filename in checkpoints/*; do
    echo $(basename $filename .pt)
    CUDA_VISIBLE_DEVICES='' fairseq-generate data-bin/wmt22.sep.tokenized.fr-de \
        --path $filename \
        --batch-size 256 --beam 5\
        --results-path evaluation_generate/$(basename $filename .pt)
done
