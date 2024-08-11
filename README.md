# Investigating the Learning Dynamics of Conditional Language Models

This is the project repository of my Master's Thesis on: Investigating the Learning Dynamics of Conditional Language Models

## Model training

The scripts for model trianing can be found in the [/scripts](scripts/) folder.
Training scripts for specific model configurations are found in [/models/tl](models/tl).

Scripts for data preprocessing can be found in [/data/tl](/data/tl).


### Model translations

Translations have been generated using [/compt/bleu/generate_test.sh](/compt/bleu/generate_test.sh).

### KL divergence

To compute the KL divergence first install Fairseq provided in [/alti/fairseq](/alti/fairseq).
KL divergence can be computed using [/kl/test_on_time.sh](/kl/test_on_time.sh). 

### ALTI+

Scirpts for alti+ computation are provided in [alti/transformer-contribuions-nmt-v2](alti/transformer-contribuions-nmt-v2).
Run [main.py](alti/transformer-contribuions-nmt-v2/main.py) to comptue the alti contirbuions over the course of training.

### LRP

Our implementation of LRP in Fairseq can be found in [/lrp/lrp_fairseq](/lrp/lrp_fairseq).

### Hallucination metrics

#### LaBSE

The python script we used to compute LaBSE cosine similairty can be found at [/comp/hallucinations/labse/labse.py](/comp/hallucinations/labse/labse.py).

#### Token hallucination metric

For the model-based token hallucination metric, clone and install the repository from the [project repository](https://github.com/violet-zct/fairseq-detect-hallucination).
Download the pretrained XSum model provided.
We used [/halu/fairseq-detect-hallucination/test/run_test.sh](/halu/fairseq-detect-hallucination/test/run_test.sh) for the computation of the token hallucination ratio.



