# ATIS - 2

### Dataset
ATIS2 contains approximately 15,000 utterances from 450 participants in the ATIS (Air Travel Information Services) collection. The ATIS collection was developed to support the research and development of speech understanding systems. Participants were presented with various hypothetical travel planning scenarios and asked to solve them by interacting with partially or completely automated ATIS systems.

Every line of the dataset contains a sentence its tags followed by the intent of the user in the sentence.
Example :- BOS american flights from chicago to los angeles morning EOS	O B-airline_name O O B-fromloc.city_name O B-toloc.city_name I-toloc.city_name B-depart_time.period_of_day atis_flight

## Files Information

### SLOT Tagging 
* Type - Jupyter Notebook
* Evaluator - Micro F1 = 97%, Macro F1 = 62%
* Description - Contains Entity tagging model training code on LSTM and BILSTM
