grep "| valid |" $SCRATCH/ma/tm_val/train_staged_tm.log > valid.txt
cut -d "|" -f 6 valid.txt | cut -c7- > loss.tmp
cut -d "|" -f 12 valid.txt | cut -c14- > step.tmp
paste loss.tmp step.tmp > valid_list.txt
rm loss.tmp
rm step.tmp

grep "| train_inner |" $SCRATCH/ma/tm_val/train_staged_tm.log > train.txt
cut -d "," -f 1 train.txt | cut -c74- > loss.tmp
cut -d "," -f 8 train.txt | cut -c14- > step.tmp
paste loss.tmp step.tmp > train_list.txt
rm loss.tmp
rm step.tmp
