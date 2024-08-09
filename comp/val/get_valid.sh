file=~/ma/models/tl/wmt22frde_subset/wmt_big/train_staged_tm.log
#file=tmp.log
grep "| valid |" $file > valid.txt
cut -d "|" -f 6 valid.txt | cut -c7- > loss.tmp
cut -d "|" -f 12 valid.txt | cut -c14- > step.tmp
paste loss.tmp step.tmp > valid_list.txt
rm loss.tmp
rm step.tmp

grep "| train_inner |" $file > train.txt
cut -d "," -f 1 train.txt | cut -d "=" -f 2 > loss.tmp
cut -d "," -f 8 train.txt | cut -d "=" -f 2 > step.tmp
paste loss.tmp step.tmp > train_list.txt
rm loss.tmp
rm step.tmp
