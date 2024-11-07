# Investigating the Learning Dynamics of Conditional Language Models

This is the project repository of my Master's Thesis on: Investigating the Learning Dynamics of Conditional Language Models 
- paper verlinken
- Hier geben wir grobe anweisungen, wo die verschiedenen skripte sind etc.

## Model training
- beschreiben was für modelle wir trainieren / was für modelle dort sind.
- Daten beschreiben / Datasets
- beschreiben, was für framework, wenn ich etwsas als vorlage benutze oder so?
  
The scripts for model training can be found in the [/scripts](scripts/) folder.
Training scripts for specific model configurations are found in [/models/tl](models/tl).

Scripts for data preprocessing can be found in [/data/tl](/data/tl).

### Model translations

Translations have been generated using [/compt/bleu/generate_test.sh](/compt/bleu/generate_test.sh).
- vlt. etwas details zu generation?

### KL divergence

To compute the KL divergence first install Fairseq provided in [/alti/fairseq](/alti/fairseq).
KL divergence can be computed using [/kl/test_on_time.sh](/kl/test_on_time.sh). 
Hier mehr details, modelle so wie zuvor benutzen, fairseq muss eh schon drauf sein. 
Klar machen, dass wir die gleichen modelle einfach benutzen können mit diesem skript

### ALTI+

Scirpts for alti+ computation are provided in [alti/transformer-contribuions-nmt-v2](alti/transformer-contribuions-nmt-v2).
Run [main.py](alti/transformer-contribuions-nmt-v2/main.py) to comptue the alti contirbuions over the course of training.
vlt. mehr details aber generell ok: Wir generieren ALTI+ scores mit dem skript, 
evtl. die änderungen von alti+ auch vorschlagen?

### LRP

Our implementation of LRP in Fairseq can be found in [/lrp/lrp_fairseq](/lrp/lrp_fairseq).
Hier viel mehr detais:
- Wir haben lrp struktur übernommen von ...
- Code wurde grösstenteils übernommen und in pytorch umgesetzt
- Skritp zum generieren der contributions findet man in ...
  
### Hallucination metrics

#### LaBSE

The python script we used to compute LaBSE cosine similairty can be found at [/comp/hallucinations/labse/labse.py](/comp/hallucinations/labse/labse.py).
actually probably enough. maybe details on installing labse.

#### Token hallucination metric

For the model-based token hallucination metric, clone and install the repository from the [project repository](https://github.com/violet-zct/fairseq-detect-hallucination).
Download the pretrained XSum model provided.
We used [/halu/fairseq-detect-hallucination/test/run_test.sh](/halu/fairseq-detect-hallucination/test/run_test.sh) for the computation of the token hallucination ratio.
- etwas ausführen aber generell ok...


