# Investigating the Learning Dynamics of Conditional Language Models

Project repository for my Master's thesis: [Link](https://www.research-collection.ethz.ch/handle/20.500.11850/697969).

## Model Setup

Our experiments are conducted using PyTorch and the [Fairseq](https://github.com/facebookresearch/fairseq) framework.

To reproduce results on a CPU, use the following setup:
- Python 3.8.20
- PyTorch 1.9.0
- Torchvision 0.10.0
  
Original experiments were conducted on a GPU with:
- PyTorch 1.2.0
- Torchvision 0.4.0 
- CUDA 10.0.130
- CUDNN 7.5

We provide a modified version of Fairseq in the [/alti/fairseq](/alti/fairseq) directory, which can be installed as follows:

```bash
cd alti/fairseq
pip install --editable ./
```


## Data

We use various translation datasets, which can be downloaded from their respective sources:

- **IWSLT14 German-English (iwslt14deen):** [Link](https://wit3.fbk.eu/2014-01)
- **WMT22 French-German (wmt22frde) & German-English (wmt22deen):** [Link](https://www.statmt.org/wmt22/translation-task.html)

Preprocessing scripts are available in [/data/tl](/data/tl). Validation and test data can be found in the respective directories.

### Data Preparation

- For **iwslt14deen**, modify [prepare-iwslt14-sep.sh](/data/tl/iwslt14deen/sentp/prepare-iwslt14-sep.sh) and run:
  ```bash
  bash prepare-iwslt14-sep.sh
  ```
- For **wmt22deen** and **wmt22frde**, download the data from the respective websites and preprocess using the adapted preparation scripts.
- For **wmt22frde subset experiments**, we use a random subset of 20M lines and apply the same preprocessing script.


## Model Training

We train models using the [Fairseq](https://github.com/facebookresearch/fairseq) framework. Scripts for data preparation and model training are available in [/scripts](/scripts). Specific training configurations are located in [/models/tl](/models/tl).
To train a translation model:

```bash
bash train_staged_tm.sh
```

Training checkpoints are stored based on the schedule in the training script.

To train a language model:

```bash
bash train_staged_lm_trunc.sh
```

### Model Translations

We generate translations using the script in [/comp/bleu/generate_test.sh](/comp/bleu/generate_test.sh). Adapt the file for your desired model and dataset, then run:

```bash
bash generate_test.sh
```

Results are stored in the respective model directories under the `evaluation_generate` folder.

## Metrics

### KL Divergence
We measure KL Divergence between translation models and unigram/bigram distributions of training data, as well as between translation and language models trained on the same datasets.
To compute the unigram and bigram distributions of the training data adapt [kl/data/distribution.py](/kl/data/distribution.py) for the respective dataset and run:
```bash
python distribution.py
```
To compute the KL divergences for a specific translation and language model pair, adapt [kl/test_on_time.sh](/kl/test_on_time.sh) and [kl/kl_on_time_efficient.py](kl/kl_on_time_efficient.py) to match the corresponding paths, data and models and run:
```bash
bash test_on_time.sh
```
Results are saved as `test_kl_dict.pkl`, `test_kl_dict_unigram.pkl`, and `test_kl_dict_bigram.pkl`.
To generate plots, place the files into the corresponding folder in [kl/data/tl/](kl/data/tl/) and run:
```bash
python plot_all.py
```
For a specific model:
``` bash
python plot_results.py iwlst14deen/iwslt
```
Plots are saved in [kl/plot](kl/plot).

### ALTI+

We compute source and target prefix contributions over training using [ALTI+](https://github.com/mt-upc/transformer-contributions-nmt). Scripts for ALTI+ computation are in [/alti/transformer-contributions-nmt-v2](/alti/transformer-contributions-nmt-v2).
To compute ALTI+ contributions adapt and run ```main.py```:
```bash
python main.py
```
Results for a specific model can be plotted by:
```bash 
python plot.py tl/iwslt14deen/iwslt
```
Plots in the thesis were generated using [plot_labse.py](/alti/transformer-contribuions-nmt-v2/plot_labse.py).

### LRP

We adapt a source attribution method based on [layer-wise relevance propagation](https://github.com/lena-voita/the-story-of-heads) for Fairseq/PyTorch to analyze source and target prefix contributions. Our LRP implementation is in [/lrp/lrp_fairseq](/lrp/lrp_fairseq). 
Adapt and run [/lrp/lrp_fairseq/main.py](/lrp/lrp_fairseq/main.py) to compute the evolution of LRP contributions over the course of training:
```bash
python main.py
```
For plotting:
```bash
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

