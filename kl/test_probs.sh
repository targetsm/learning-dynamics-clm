#!/bin/bash
source $HOME/ma/venv_new/bin/activate
cd $SCRATCH/ma/tm
mkdir -p evaluation
mkdir -p evaluation/probs
for filename in checkpoints/*; do
    echo $filename
    fairseq-generate data-bin/iwslt14.sep.tokenized.de-en \
        --path $filename \
        --batch-size 128 --beam 5 \
	--results-path evaluation/$(basename $filename .pt) \
        --score-reference --no-progress-bar > evaluation/probs/$(basename $filename .npy)
done

cd $SCRATCH/ma/lm_trunc
mkdir -p evaluation
mkdir -p evaluation/probs

for filename in checkpoints/*; do
    echo $filename
    fairseq-generate data-bin/iwslt14.sep.tokenized.de-en \
        --path $filename \
        --batch-size 128 --beam 5 \
        --results-path evaluation/$(basename $filename .pt) \
        --score-reference --no-progress-bar > evaluation/probs/$(basename $filename .npy)
done

