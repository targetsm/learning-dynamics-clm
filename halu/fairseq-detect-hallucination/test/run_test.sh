
for f in ../../../models/tl/iwslt14deen/iwslt/tm/evaluation_generate/*/*.txt
do
	echo $f
	grep "^T" $f | cut -f 2 > target.en
	grep "^H" $f | cut -f 3 > pred.en
	paste target.en pred.en | shuf | head -n 1000 > paste.tmp
	cut -f 1 paste.tmp > data/target.en
	sed -i -e 's/ //g' data/target.en 
	sed -i -e "s/&apos;/\'/" data/target.en
	sed -i -e 's/▁/ /g' data/target.en
	cut -f 2 paste.tmp > data/pred.en
	sed -i -e 's/ //g' data/pred.en
	sed -i -e "s/&apos;/\'/" data/pred.en
        sed -i -e 's/▁/ /g' data/pred.en
	rm target.en pred.en paste.tmp
	CUDA_VISIBLE_DEVICES="" python predict_hallucination_mt.py $f
done
