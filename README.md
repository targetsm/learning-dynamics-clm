# Investigating the Learning Dynamics of Conditional Language Models

Project repository for my Master's thesis. The thesis can be accessed at: [https://www.research-collection.ethz.ch/handle/20.500.11850/697969](https://www.research-collection.ethz.ch/handle/20.500.11850/697969).

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

### Model Training

TODO: Describe different models used: iwslt, wmt, wmt_big

We train various models using the [Fairseq](https://github.com/facebookresearch/fairseq) framework. Scripts for data preparation and model training can be found in the [/scripts](/scripts) folder. Training scripts for specific model configurations are in [/models/tl](/models/tl).

To train a translation model, modify the specific training file to match desired data, model and hyperparameters. Then run:
```
bash train_staged_tm.sh
```
Checkpoints are found in checkpoints. checkpoints are stored according to schedule in training file.

Similarly langauge models can be trained by adapting and running:
```
bash train_staged_lm_trunc.sh
```

### Model Translations

We generate translations using the script in [/comp/bleu/generate_test.sh](/compt/bleu/generate_test.sh).
Adapt the file to desired model and data and run:

```
bash generate_test.sh
```

Output stored in folder evaluation_generate in subfolder of model.

## Metrics

### KL Divergence

To compute the KL divergence between translation and language models.
We train a language model and a translation model on the same data as described above. 
We compute the kl divergence between tranlsation models and unigram and bigram distributions of the training data as well as translation models and language models Fitted to the same datasets.

To compute the unigram and bigram distributions of the training data adapt [kl/data/distribution.py](/kl/data/distribution.py) to fit the dataset and setup and run:
```
python distribution.py
```

To compute the KL divergences for a specific translation model adapt [kl/test_on_time.sh](/kl/test_on_time.sh) and [kl/kl_on_time_efficient.py](kl/kl_on_time_efficient.py) to the corresponding paths, data and models and run:

```
bash test_on_time.sh
```

Results are stored in test_kl_dict.pkl, test_kl_dict_unigram.pkl and test_kl_dict_bigram.pkl.
Place the files into the corresponding folder in kl/data/tl/
Plots can be generated running 
```
python plot_all.py
```
plots for individual models can be generated using:

```
python plot_restults.py iwlst14deen/iwslt
```

The results are stored in the kl/plot folder.

### ALTI+

We compute source and target prefix contributions using [ALTI+](https://github.com/mt-upc/transformer-contributions-nmt). Scripts for ALTI+ computation are in [alti/transformer-contribuions-nmt-v2](/alti/transformer-contribuions-nmt-v2). Adapt and run [main.py](/alti/transformer-contribuions-nmt-v2/main.py) to compute the evolution of ALTI+ contributions over the course of training.

```
python main.py
```

Results can be plottet using 
``` python plot.py tl/iwslt14deen/iwslt ```
for a single model.

The plots in the thesis can be generated using /alti/transformer-contribuions-nmt-v2/plot_labse.py

### LRP

We adapt a [source attribution method](https://github.com/lena-voita/the-story-of-heads) based on layer-wise relevance propagation for Fairseq/PyTorch to compute source and target prefix contributions. Our implementation of LRP in Fairseq is in [/lrp/lrp_fairseq](/lrp/lrp_fairseq). Adapt and run [/lrp/lrp_fairseq/main.py](/lrp/lrp_fairseq/main.py) to compute the evolution of LRP contributions over the course of training.
```
python main.py
```
To plot results run:
```
python plot.py plt/iwslt14deen/iwslt
```
Further plotting can be run using:
```
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
Use plot.py to generate resutls for a specific model and plot_all.py to generate plots for multiple models.

### Token Hallucination Metric

For the model-based token hallucination metric, clone and install the repository from the [project repository](https://github.com/violet-zct/fairseq-detect-hallucination). Download the pre-trained XSum model provided. We used [/halu/fairseq-detect-hallucination/test/run_test.sh](/halu/fairseq-detect-hallucination/test/run_test.sh) and [/halu/fairseq-detect-hallucination/test/predict_hallucination_mt.py](/halu/fairseq-detect-hallucination/test/run_test.sh) to compute the token hallucination ratio.

