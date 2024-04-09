mkdir -p $HOME/ma/kl/scores
mkdir -p $HOME/ma/kl/scores/tm
cd $SCRATCH/ma/tm/evaluation
for ckpt in ./*; do
    grep "^P" $SCRATCH/ma/tm/evaluation/$ckpt/generate-test.txt | cut -c3- > tm.val
    grep "^D" $SCRATCH/ma/tm/evaluation/$ckpt/generate-test.txt | cut -c3- > tm.txt
    paste tm.txt tm.val > $HOME/ma/kl/scores/tm/$ckpt
    rm tm.txt tm.val
done

mkdir -p $HOME/ma/kl/scores/lm_trunc
cd $SCRATCH/ma/lm_trunc/evaluation
for ckpt in ./*; do
    grep "^P" $SCRATCH/ma/lm_trunc/evaluation/$ckpt/generate-test.txt | cut -c3- > lm_trunc.val
    grep "^D" $SCRATCH/ma/lm_trunc/evaluation/$ckpt/generate-test.txt | cut -c3- > lm_trunc.txt
    paste lm_trunc.txt lm_trunc.val > $HOME/ma/kl/scores/lm_trunc/$ckpt
    rm lm_trunc.txt lm_trunc.val
done
