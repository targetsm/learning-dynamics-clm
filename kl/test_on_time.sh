#!/bin/bash
ulimit -m 20000000
source $HOME/ma/alti/venv_alti/bin/activate
workdir=/local/home/ggabriel/ma/models/tl/wmt22frde
cd $workdir/tm/checkpoints

#if [ ! -d analysis ]; then
#   mkdir -p analysis
#   cp checkpoint_*_*.pt analysis
#   cd analysis
#   for filename in ./*; do echo $filename; mv $filename ${filename##*_}; echo ${filename##*_}; done
#fi

#cd $workdir/lm/checkpoints
#if [ ! -d analysis ]; then
#   mkdir -p analysis
#   cp checkpoint_*_*.pt analysis
#   cd analysis
#   for filename in ./*; do mv $filename ${filename##*_}; done
#fi

cd $workdir
mkdir -p evaluation

for filename in $workdir/tm/checkpoints/analysis/*; do
    f=$(basename $filename .pt)
    echo $filename
    echo $f
    CUDA_VISIBLE_DEVICES='' fairseq-generate tm/data-bin/wmt22.sep.tokenized.fr-de \
        --path tm/checkpoints/analysis/$f.pt \
        --batch-size 256 --beam 1\
	--results-path evaluation/tm \
        --score-reference
    CUDA_VISIBLE_DEVICES='' fairseq-generate lm/data-bin/wmt22.sep.tokenized.fr-de \
        --path lm/checkpoints/analysis/$f.pt \
        --batch-size 256 --beam 1 \
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
    python -u kl_on_time_efficient.py $f
    cd $workdir
    rm -rf evaluation/*
done

