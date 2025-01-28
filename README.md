# Investigating the Learning Dynamics of Conditional Language Models

Project repository for my Master's Thesis [Link](https://www.research-collection.ethz.ch/handle/20.500.11850/697969).

## Model Setup

Our experiments rely on PyTorch and the [Fairseq](https://github.com/facebookresearch/fairseq) framework. We provide an adapted version of Fairseq under [/alti/fairseq](/alti/fairseq).

### Data

We use various translation datasets in our experiments, which can be downloaded from their respective websites. The datasets include:

- IWSLT14 German to English MT Track (iwslt14deen), available at [Link](https://wit3.fbk.eu/2014-01)
- WMT22 French to German (wmt22deen) and WMT22 German to English (wmt22deen), both available at [Link](https://www.statmt.org/wmt22/translation-task.html)

We preprocess the data using scripts found in [/data/tl](/data/tl).

### Model Training

We train various translation models using the [Fairseq](https://github.com/facebookresearch/fairseq) framework. Scripts for data preparation and model training are in the [/scripts](/scripts) folder. Training scripts for specific model configurations are in [/models/tl](/models/tl).

### Model Translations

We generate translations using the script in [/compt/bleu/generate_test.sh](/compt/bleu/generate_test.sh).

## Experiments

### KL Divergence

To compute the KL divergence between translation and language models, first install Fairseq provided in [/alti/fairseq](/alti/fairseq). We compute the KL divergence using [/kl/test_on_time.sh](/kl/test_on_time.sh).

### ALTI+

We compute source and target prefix contributions using [ALTI+](https://github.com/mt-upc/transformer-contributions-nmt). Scripts for ALTI+ computation are in [alti/transformer-contribuions-nmt-v2](/alti/transformer-contribuions-nmt-v2). Run [main.py](/alti/transformer-contribuions-nmt-v2/main.py) to compute the evolution of ALTI+ contributions over the course of training.

### LRP

We adapt a [source attribution method](https://github.com/lena-voita/the-story-of-heads) based on layer-wise relevance propagation for Fairseq/PyTorch to compute source and target prefix contributions. Our implementation of LRP in Fairseq is in [/lrp/lrp_fairseq](/lrp/lrp_fairseq). Run [/lrp/lrp_fairseq/main.py](/lrp/lrp_fairseq/main.py) to compute the evolution of LRP contributions over the course of training.

### Hallucination Metrics

### LaBSE

Install [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) as described. The Python script we used to compute LaBSE cosine similarity is at [/comp/hallucinations/labse/labse.py](/comp/hallucinations/labse/labse.py).

### Token Hallucination Metric

For the model-based token hallucination metric, clone and install the repository from the [project repository](https://github.com/violet-zct/fairseq-detect-hallucination). Download the pre-trained XSum model provided. We used [/halu/fairseq-detect-hallucination/test/run_test.sh](/halu/fairseq-detect-hallucination/test/run_test.sh) to compute the token hallucination ratio.
