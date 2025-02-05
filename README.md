# Investigating the Learning Dynamics of Conditional Language Models

Project repository for my Master's Thesis. The thesis can be accessed [here](https://www.research-collection.ethz.ch/handle/20.500.11850/697969).

## Model Setup

Our experiments rely on PyTorch and the [Fairseq](https://github.com/facebookresearch/fairseq) framework. 
My Setup:
- PyTorch v1.4.0
- CUDA 10.0.130
- CUDNN 7.5

We provide an adapted version of Fairseq under [/alti/fairseq](/alti/fairseq).
Can be installed by running:

``` 
cd alti/fairseq
pip install --editable ./
```

### Data

We use various translation datasets in our experiments, which can be downloaded from their respective websites. The datasets include:

- IWSLT14 German to English MT Track (iwslt14deen), available at [Link](https://wit3.fbk.eu/2014-01)
- WMT22 French to German (wmt22deen) and WMT22 German to English (wmt22deen), both available at [Link](https://www.statmt.org/wmt22/translation-task.html)

We preprocess the data using scripts found in [/data/tl](/data/tl). TODO: add more details

### Model Training

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

## Experiments

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

### ALTI+

We compute source and target prefix contributions using [ALTI+](https://github.com/mt-upc/transformer-contributions-nmt). Scripts for ALTI+ computation are in [alti/transformer-contribuions-nmt-v2](/alti/transformer-contribuions-nmt-v2). Run [main.py](/alti/transformer-contribuions-nmt-v2/main.py) to compute the evolution of ALTI+ contributions over the course of training.

TODO: add commands

### LRP

We adapt a [source attribution method](https://github.com/lena-voita/the-story-of-heads) based on layer-wise relevance propagation for Fairseq/PyTorch to compute source and target prefix contributions. Our implementation of LRP in Fairseq is in [/lrp/lrp_fairseq](/lrp/lrp_fairseq). Run [/lrp/lrp_fairseq/main.py](/lrp/lrp_fairseq/main.py) to compute the evolution of LRP contributions over the course of training.

TODO: add commands for execution

### Hallucination Metrics

### LaBSE

Install [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) as described. The Python script we used to compute LaBSE cosine similarity is at [/comp/hallucinations/labse/labse.py](/comp/hallucinations/labse/labse.py).

TODO: add commands

### Token Hallucination Metric

For the model-based token hallucination metric, clone and install the repository from the [project repository](https://github.com/violet-zct/fairseq-detect-hallucination). Download the pre-trained XSum model provided. We used [/halu/fairseq-detect-hallucination/test/run_test.sh](/halu/fairseq-detect-hallucination/test/run_test.sh) to compute the token hallucination ratio.

TODO: add commands, also probably more details on the metric?
