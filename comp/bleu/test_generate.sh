#!/bin/bash
source $HOME/ma/alti/venv_alti/bin/activate
cd $SCRATCH/ma/tm_val
mkdir -p evaluation_generate

for filename in checkpoints/analysis/*; do
    echo $filename
    fairseq-generate data-bin/iwslt14.sep.tokenized.de-en \
        --path $filename \
        --batch-size 128 --beam 5\
        --results-path evaluation_generate/$(basename $filename .pt)
done
