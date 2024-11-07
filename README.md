# Investigating the Learning Dynamics of Conditional Language Models

This is the project repository of my Master's Thesis on: [Investigating the Learning Dynamics of Conditional Language Models](https://www.research-collection.ethz.ch/handle/20.500.11850/697969) 

## Data
We use different datasets:
- iwslt14deen
- wmt22deen
- wmt22frde

TODO Data perparation

## Model training
We train different translation models using fairseq on several datasets.

TODO:
- describe what models we train / what models are there.
- Describe data / datasets
- framework we use
  
The scripts for model training can be found in the [/scripts](scripts/) folder.
Training scripts for specific model configurations are found in [/models/tl](models/tl).

Scripts for data preprocessing can be found in [/data/tl](/data/tl).

### Model translations

Translations have been generated using [/compt/bleu/generate_test.sh](/compt/bleu/generate_test.sh).
TODO: Details about generation?

### KL divergence

To compute the KL divergence first install Fairseq provided in [/alti/fairseq](/alti/fairseq).
KL divergence can be computed using [/kl/test_on_time.sh](/kl/test_on_time.sh). 
TODO:
- Here more details, use models as before, remove fairseq here
- Make it clear that we can just use the same models with this script

### ALTI+

Scirpts for alti+ computation are provided in [alti/transformer-contribuions-nmt-v2](alti/transformer-contribuions-nmt-v2).
Run [main.py](alti/transformer-contribuions-nmt-v2/main.py) to comptue the alti contirbuions over the course of training.
TODO:
- Maybe more details but generally ok: We generate ALTI+ scores with the script,
- Possibly also suggest the changes of alti+?

### LRP

Our implementation of LRP in Fairseq can be found in [/lrp/lrp_fairseq](/lrp/lrp_fairseq).
Here much more detais:
- We have taken over lrp structure from ...
- Code was mostly taken over and converted into pytorch
- Skritp to generate the contributions can be found in ...
  
### Hallucination metrics

#### LaBSE

The python script we used to compute LaBSE cosine similairty can be found at [/comp/hallucinations/labse/labse.py](/comp/hallucinations/labse/labse.py).
actually probably enough. maybe details on installing labse.

#### Token hallucination metric

For the model-based token hallucination metric, clone and install the repository from the [project repository](https://github.com/violet-zct/fairseq-detect-hallucination).
Download the pretrained XSum model provided.
We used [/halu/fairseq-detect-hallucination/test/run_test.sh](/halu/fairseq-detect-hallucination/test/run_test.sh) for the computation of the token hallucination ratio.


