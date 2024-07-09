#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

#echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git ../../mosesdecoder

#echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git ../../subword-nmt

SCRIPTS=../../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=../../subword-nmt/subword_nmt
BPE_TOKENS=10000


src=fr
tgt=de
lang=fr-de
prep=wmt22.sep.tokenized.fr-de
tmp=$prep/tmp
orig=orig

#mkdir -p $orig $tmp $prep
#
#
#echo "pre-processing train data..."
#for l in $src $tgt; do
#    f=train.$l
#    tok=train.tok.$l
#
#    cat $orig/$f | \
#    grep -v '<url>' | \
#    grep -v '<talkid>' | \
#    grep -v '<keywords>' | \
#    sed -e 's/<title>//g' | \
#    sed -e 's/<\/title>//g' | \
#    sed -e 's/<description>//g' | \
#    sed -e 's/<\/description>//g' | \
#    perl $TOKENIZER -l $l > $tmp/$tok
#    echo ""
#done
#perl $CLEAN -ratio 1.5 $tmp/train.tok $src $tgt $tmp/train.clean 1 175
#for l in $src $tgt; do
#    perl $LC < $tmp/train.clean.$l > $tmp/train.$l
#done
#
#echo "pre-processing valid/test data..."
#for l in $src $tgt; do
#    f=dev.$l
#    tok=valid.tok.$l
#
#    cat $orig/$f | \
#    grep -v '<url>' | \
#    grep -v '<talkid>' | \
#    grep -v '<keywords>' | \
#    sed -e 's/<title>//g' | \
#    sed -e 's/<\/title>//g' | \
#    sed -e 's/<description>//g' | \
#    sed -e 's/<\/description>//g' | \
#    perl $TOKENIZER -l $l > $tmp/$tok
#    echo ""
#done
#for l in $src $tgt; do
#    perl $LC < $tmp/valid.tok.$l > $tmp/valid.$l
#done
#
#for l in $src $tgt; do
#    f=test.$l
#    tok=test.tok.$l
#
#    cat $orig/$f | \
#    grep -v '<url>' | \
#    grep -v '<talkid>' | \
#    grep -v '<keywords>' | \
#    sed -e 's/<title>//g' | \
#    sed -e 's/<\/title>//g' | \
#    sed -e 's/<description>//g' | \
#    sed -e 's/<\/description>//g' | \
#    perl $TOKENIZER -l $l > $tmp/$tok
#    echo ""
#done
#for l in $src $tgt; do
#    perl $LC < $tmp/test.tok.$l > $tmp/test.$l
#done
#
#for l in $src $tgt; do
#    #for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
#    #fname=${o##*/}
#    #f=$tmp/${fname%.*} 
#    echo $o $f
#    grep '<seg id' $o | \
#        sed -e 's/<seg id="[0-9]*">\s*//g' | \
#        sed -e 's/\s*<\/seg>\s*//g' | \
#        sed -e "s/\’/\'/g" | \
#    perl $TOKENIZER -l $l | \
#    perl $LC > $f
#    echo ""
#    done
#done
#

#echo "creating train, valid, test..."
#for l in $src $tgt; do
#    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
#    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l
#
#    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
#        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
#        $tmp/IWSLT14.TED.tst2010.de-en.$l \
#        $tmp/IWSLT14.TED.tst2011.de-en.$l \
#        $tmp/IWSLT14.TED.tst2012.de-en.$l \
#        > $tmp/test.$l
#done
#

SCRIPTS=~/ma/alti/fairseq/scripts
SPM_TRAIN=$SCRIPTS/spm_train.py
SPM_ENCODE=$SCRIPTS/spm_encode.py

# learn BPE with sentencepiece
echo "learning separate BPE"
for l in $src $tgt; do
    TRAIN=$tmp/train.$l
    BPE_CODE=$prep/code.$l
    python $SPM_TRAIN \
        --input=$TRAIN \
        --model_prefix=$BPE_CODE \
        --vocab_size=$BPE_TOKENS \
        --character_coverage=1.0 \
        --model_type=bpe
done
# encode train/valid/test
echo "encoding train/valid with learned BPE..."

for LANG in $src $tgt; do
    for f in train.$LANG valid.$LANG test.$LANG; do
    	python $SPM_ENCODE \
           --model $prep/code.$LANG.model \
           --output_format=piece \
           --inputs $tmp/$f \
           --outputs $prep/$f 
           #--min-len $TRAIN_MINLEN --max-len $TRAIN_MAXLEN
    done
done

#
#for l in $src $tgt; do
#    TRAIN=$tmp/train.$l
#    BPE_CODE=$prep/code.$l
#    
#    echo "learn_bpe.py on ${TRAIN}..."
#    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
#done
#
#for L in $src $tgt; do
#    for f in train.$L valid.$L test.$L; do
#        echo "apply_bpe.py to ${f}..."
#        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
#    done
#done
