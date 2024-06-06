mkdir -p log_probs
for ckpt in $SCRATCH/ma/tm_val/evaluation_generate/*; do
	grep "H" $ckpt/generate-test.txt  | cut -f 2 > log_probs/$(basename $ckpt).txt
done

mkdir -p log_probs_test
for ckpt in $SCRATCH/ma/tm/evaluation/*; do
        grep "H" $ckpt/generate-test.txt  | cut -f 2 > log_probs_test/$(basename $ckpt).txt
done

