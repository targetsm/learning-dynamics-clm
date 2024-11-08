# Investigating the Learning Dynamics of Conditional Language Models

This is the project repository of the Master's Thesis on: [Investigating the Learning Dynamics of Conditional Language Models](https://www.research-collection.ethz.ch/handle/20.500.11850/697969) 

## Model setup

The experiments rely on PyTorch and the [Fairseq](https://github.com/facebookresearch/fairseq) framework.
A altered version of Fairseq is provided under [/alti/fairseq](/alti/fairseq), with necessary adaptations for our experiments.

[comment]: # (Specify what libraries exactly are necessary? all of them)

### Data
In our experiments we use different translation datasets. 
The datasets can be donwloaded from the respective Websites. 
We use the following datasets:
- IWSLT14 German to English MT Track (iwslt14deen) available at [Link](https://wit3.fbk.eu/2014-01)
- WMT22 French to German (wmt22deen) and WMT22 German to English (wmt22deen) both available at [Link](https://www.statmt.org/wmt22/translation-task.html)

We preprocess the data using scripts found in [/data/tl](/data/tl).

[comment]: # (Maybe add data subset generation?)

### Model training

We train different translation models using the [Fairseq](https://github.com/facebookresearch/fairseq) framework.
The scripts for data preparation and model training can be found in the [/scripts](scripts/) folder.
Training scripts for specific model configurations are found in [/models/tl](models/tl).

### Model translations

We generate translations using the script provided in [/compt/bleu/generate_test.sh](/compt/bleu/generate_test.sh).

## Experiments

### KL divergence

To compute the KL divergence between translation and language models first install Fairseq provided in [/alti/fairseq](/alti/fairseq).
We compute the KL divergence using [/kl/test_on_time.sh](/kl/test_on_time.sh).

### ALTI+

We compute source and target prefix contributions using [ALTI+](https://github.com/mt-upc/transformer-contributions-nmt).
Scirpts for ALTI+ computation are provided in [alti/transformer-contribuions-nmt-v2](alti/transformer-contribuions-nmt-v2).
Run [main.py](alti/transformer-contribuions-nmt-v2/main.py) to comptue the evolution ALTI+ contribuitons over the course of training.

### LRP

We adapt a [source attribution method](https://github.com/lena-voita/the-story-of-heads) based on layerwise relevance propagation for Fairseq/PyTorch to compute source and target prefix contributions.
Our implementation of LRP in Fairseq can be found in [/lrp/lrp_fairseq](/lrp/lrp_fairseq).
Run [/lrp/lrp_fairseq/main.py](/lrp/lrp_fairseq/main.py) to compute the evolution of LRP contributions over the course.

[comment]: # (link plotting scripts)

### Hallucination metrics

#### LaBSE

The python script we used to compute [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) cosine similairty can be found at [/comp/hallucinations/labse/labse.py](/comp/hallucinations/labse/labse.py).

#### Token hallucination metric

For the model-based token hallucination metric, clone and install the repository from the [project repository](https://github.com/violet-zct/fairseq-detect-hallucination).
Download the pretrained XSum model provided.
We used [/halu/fairseq-detect-hallucination/test/run_test.sh](/halu/fairseq-detect-hallucination/test/run_test.sh) for the computation of the token hallucination ratio.


