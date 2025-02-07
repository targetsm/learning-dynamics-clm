# Investigating the Learning Dynamics of Conditional Language Models

Project repository for my Master's thesis [Link](https://www.research-collection.ethz.ch/handle/20.500.11850/697969)

## Model Setup

Our experiments rely on PyTorch and the [Fairseq](https://github.com/facebookresearch/fairseq) framework. 

Setup under which the results can be recreated on cpu:
- Python 3.8.20
- PyTorch 1.9.0
- Torchvision 0.10.0
  
Original experiments were run on gpu using:
- PyTorch 1.2.0
- Torchvision 0.4.0 
- CUDA 10.0.130
- CUDNN 7.5

We provide an adapted version of Fairseq under [/alti/fairseq](/alti/fairseq).
It can be installed by running:

``` 
cd alti/fairseq
pip install --editable ./
```


### Data

We use various translation datasets in our experiments, which can be downloaded from their respective websites. The datasets include:

- IWSLT14 German to English MT Track (iwslt14deen), available at [Link](https://wit3.fbk.eu/2014-01)
- WMT22 French to German (wmt22deen) and WMT22 German to English (wmt22deen), both available at [Link](https://www.statmt.org/wmt22/translation-task.html)

We preprocess the data using scripts found in [/data/tl](/data/tl) for each corresponding datasets.
Validation and test data can be found in the respective directories.

Generating the training data:
For iwslt14deen adapt [prepare-iwslt14-sep.sh](/data/tl/iwslt14deen/sentp/prepare-iwslt14-sep.sh) to fit your setup and run ```bash prepare-iwslt14-sep.sh``` to generate the data.
For wmt22deen and wmt22frde datasets download the data as described on the respective websites. Adapt and run the respective prepare-iwslt14-sep.sh files to preprocess the datasets.
For the wmt22frde subset experiments, we use a random subset of 20M lines of the downloaded data and run the same preparation script on the subset of the data. 


### Model Training

We train various models using the [Fairseq](https://github.com/facebookresearch/fairseq) framework. Scripts for data preparation and model training can be found in the [/scripts](/scripts) folder. Training scripts for specific model configurations used in the thesis are found in [/models/tl](/models/tl).
To train a translation model, modify the specific training file to match desired data, model and hyper-parameters. Then run:
```
bash train_staged_tm.sh
```
Training checkpoints are stored according to the schedule in the training file. 
Similarly language models can be trained by adapting and running:
```
bash train_staged_lm_trunc.sh
```

### Model Translations

We generate translations using the script in [/comp/bleu/generate_test.sh](/compt/bleu/generate_test.sh).
Adapt the file to use the desired model and data and run:
```
bash generate_test.sh
```
The output is stored in the respective model directories in the ```evaluation_generate``` folder.


## Metrics

### KL Divergence
We track the KL Divergence between translation models and unigram and bigram distributions of the training data as well as translation models and language models fitted to the same datasets over the course of training.
To compute the unigram and bigram distributions of the training data adapt [kl/data/distribution.py](/kl/data/distribution.py) for the respective dataset and run:
```
python distribution.py
```
To compute the KL divergences for a specific translation and language model pair, adapt [kl/test_on_time.sh](/kl/test_on_time.sh) and [kl/kl_on_time_efficient.py](kl/kl_on_time_efficient.py) to match the corresponding paths, data and models and run:
```
bash test_on_time.sh
```
Results are stored as ```test_kl_dict.pkl```, ```test_kl_dict_unigram.pkl``` and ```test_kl_dict_bigram.pkl```.
To generate plots, place the files into the corresponding folder in [kl/data/tl/](kl/data/tl/) and run:
```
python plot_all.py
```
To generate the plot for a specific model, run:
```
python plot_results.py iwlst14deen/iwslt
```
The results are stored in [kl/plot](kl/plot).

### ALTI+

We compute source and target prefix contributions over the course of training using [ALTI+](https://github.com/mt-upc/transformer-contributions-nmt). Scripts for ALTI+ computation are found in [alti/transformer-contribuions-nmt-v2](/alti/transformer-contribuions-nmt-v2). Adapt and run [main.py](/alti/transformer-contribuions-nmt-v2/main.py) to compute the evolution of ALTI+ contributions over the course of training. 
Results for a specific model can be plotted by running ``` python plot.py tl/iwslt14deen/iwslt ```.
The plots in the thesis were generated with [plot_labse.py](/alti/transformer-contribuions-nmt-v2/plot_labse.py).

### LRP

We adapt a source attribution method based on [layer-wise relevance propagation](https://github.com/lena-voita/the-story-of-heads) for Fairseq/PyTorch to compute source and target prefix contributions. Our implementation of LRP in Fairseq is found in [/lrp/lrp_fairseq](/lrp/lrp_fairseq). Adapt and run [/lrp/lrp_fairseq/main.py](/lrp/lrp_fairseq/main.py) to compute the evolution of LRP contributions over the course of training:
```
python main.py
```
The following commands can be used for plotting:
```
python plot.py plt/iwslt14deen/iwslt
python plot_alti.py
```

## Hallucination Metrics

### LaBSE

Install [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) as described. The Python script we used to compute LaBSE cosine similarity is at [/comp/hallucinations/labse/labse.py](/comp/hallucinations/labse/labse.py).

To generate laBSE results run:
```
pip install sentence_transformers
python labse.py iwslt14deen/iwslt
```
Use ```plot.py``` to generate results for a specific model and ```plot_all.py``` to generate plots for multiple models.

### Token Hallucination Metric

For the model-based token hallucination metric, clone and install the repository from the [project repository](https://github.com/violet-zct/fairseq-detect-hallucination). 
Download the pre-trained XSum model provided. 
We used [/halu/fairseq-detect-hallucination/test/run_test.sh](/halu/fairseq-detect-hallucination/test/run_test.sh) and [/halu/fairseq-detect-hallucination/test/predict_hallucination_mt.py](/halu/fairseq-detect-hallucination/test/run_test.sh) to compute token hallucination ratios for trained models.

