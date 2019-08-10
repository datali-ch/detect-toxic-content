# Detect Toxic Content
Use NLP to detect abusive content

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

The challenge is to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate. The model(s) will hopefully help online discussion become more productive and respectful.

SETUP:

* Download data (train.csv) from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
* Download GloVe embeddings from http://nlp.stanford.edu/data/glove.6B.zip and unpack them
* Parametrize above data files (DATA_FILE, GLOVE_FILE) in config.py
* Install packages from requirements.txt
* Run script from command line:
```
python main.py --choose-model=YOUR_MODEL
```
where YOUR_MODEL is
1. for Bag of Words
2. for Latent Dirichlet Allocation
3. for Long Short-Term Memory

