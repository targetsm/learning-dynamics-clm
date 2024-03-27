mkdir -p $HOME/ma/kl/scores
cd $SCRATCH/ma/tm/evaluation
for ckpt in ./*; do
    grep "^P" $SCRATCH/ma/tm/evaluation/$ckpt/generate-test.txt | cut -c3- > tm.val
    grep "^D" $SCRATCH/ma/tm/evaluation/$ckpt/generate-test.txt | cut -c3- > tm.txt
    paste tm.txt tm.val > $HOME/ma/kl/scores/$ckpt.tm
    rm tm.txt tm.val
done

cd $SCRATCH/ma/lm_trunc/evaluation
for ckpt in ./*; do
    grep "^P" $SCRATCH/ma/lm_trunc/evaluation/$ckpt/generate-test.txt | cut -c3- > lm_trunc.val
    grep "^D" $SCRATCH/ma/lm_trunc/evaluation/$ckpt/generate-test.txt | cut -c3- > lm_trunc.txt
    paste lm_trunc.txt lm_trunc.val > $HOME/ma/kl/scores/$ckpt.lm_trunc
    rm lm_trunc.txt lm_trunc.val
done
