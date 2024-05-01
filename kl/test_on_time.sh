#!/bin/bash
source $HOME/ma/alti/venv_alti/bin/activate
cd $SCRATCH/ma/tm_val/checkpoints

if [ ! -d analysis ]; then
   mkdir -p analysis
   cp checkpoint_*_*.pt analysis
   cd analysis
   for filename in ./*; do mv $filename ${filename##*_}; done
fi

cd $SCRATCH/ma/lm_trunc_val/checkpoints
if [ ! -d analysis ]; then
   mkdir -p analysis
   cp checkpoint_*_*.pt analysis
   cd analysis
   for filename in ./*; do mv $filename ${filename##*_}; done
fi

cd $SCRATCH/ma
mkdir -p evaluation

for filename in $SCRATCH/ma/tm_val/checkpoints/analysis/*; do
    filename=$(basename $filename .pt)
    echo $filename
    fairseq-generate tm_val/data-bin/iwslt14.sep.tokenized.de-en \
        --path tm_val/checkpoints/analysis/$filename.pt \
        --batch-size 128 --beam 1\
	--results-path evaluation/tm \
        --score-reference
    fairseq-generate lm_trunc_val/data-bin/iwslt14.sep.tokenized.de-en \
        --path lm_trunc_val/checkpoints/analysis/$filename.pt \
        --batch-size 128 --beam 1 \
	--results-path evaluation/lm \
        --score-reference
    
    grep "^P" evaluation/tm/generate-test.txt | cut -c3- > tm.val
    grep "^D" evaluation/tm/generate-test.txt | cut -c3- > tm.txt
    paste tm.txt tm.val > evaluation/scores_tm.txt
    rm tm.txt tm.val

    grep "^P" evaluation/lm/generate-test.txt | cut -c3- > lm.val
    grep "^D" evaluation/lm/generate-test.txt | cut -c3- > lm.txt
    paste lm.txt lm.val > evaluation/scores_lm.txt
    rm lm.txt lm.val
    
    cd $HOME/ma/kl
    python kl_on_time.py $filename
    cd $SCRATCH/ma
    rm -rf evaluation/*
done

