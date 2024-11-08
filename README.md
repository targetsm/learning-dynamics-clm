# Investigating the Learning Dynamics of Conditional Language Models

This is the project repository of my Master's Thesis on: [Investigating the Learning Dynamics of Conditional Language Models](https://www.research-collection.ethz.ch/handle/20.500.11850/697969) 

## Data
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


