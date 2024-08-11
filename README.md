# Investigating the Learning Dynamics of Conditional Language Models

This is the project repository of my Master's Thesis on: Investigating the Learning Dynamics of Conditional Language Models

## Model training

The scripts for model trianing can be found in the [/scripts](scripts/) folder.
Training scripts for specific model configurations are found in [/models/tl](models/tl).

Scripts for data preprocessing can be found in [/data/tl/](/data/tl).


### Model translations

Translations have been generated using [/compt/bleu/generate_test.sh](/compt/bleu/generate_test.sh).

### KL divergence

To compute the KL divergence first install fairseq provided in [/alti/fairseq](/alti/fairseq).
KL divergence can be computed using [/kl/test_on_time.sh](/kl/test_on_time.sh). 

### ALTI+

Scirpts for alti+ computattion are provdied in [alti/transformer-contribuions-nmt-v2](alti/transformer-contribuions-nmt-v2)
Run [main.py](alti/transformer-contribuions-nmt-v2/main.py) to comptue the alti contirbuions over the course of training.

### LRP

### Hallucination metrics

### Token hallucination metric

### LaBSE
