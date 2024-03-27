#!/bin/bash
source $HOME/ma/venv_new/bin/activate
cd $SCRATCH/ma/test
mkdir -p evaluation

for filename in checkpoints/steps/*.pt; do
    echo $filename
    fairseq-generate data-bin/iwslt14.tokenized.de-en \
        --path $filename \
        --batch-size 128 --beam 5 --remove-bpe \
	--results-path evaluation/$(basename $filename .pt)
done
