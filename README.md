# Investigating the Learning Dynamics of Conditional Language Models

This is the project repository of my Master's Thesis on: Investigating the Learning Dynamics of Conditional Language Models

## Model training
We train several translation models on differet datasets.
The respective datasets can be downloaded from :
...

The data can be preprocessed using the rspecitve scripts in data/tl/

Training scripts for the different files can be found in the model folder

### Model translations

Translations have been generated using compt/bleu/generate_test.sh

### KL divergence

To compute the KL divergence first install fairseq provided in alti/fairseq
Then run scirpt to generate the KL divergence.

### ALTI+

Scirpts for alti+ computattion are provdied in alti/transformer-contribuions-nmt-v2
Run main.py to comptue the alti contirbuions over the course of training.

### LRP

### Hallucination metrics

### Token hallucination metric

### LaBSE
