# Variational Attention
![](https://img.shields.io/badge/python-3.6-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.3.0-orange.svg)
Implentation of 'Variational Attention for Sequence to Sequence Models' in tensorflow.

## Overview
This package consists of 3 models, each of which have been organized into separate folders:
1. Deterministic encoder-decoder with deterministic attention (`ded_detAttn`)
2. Variational encoder-decoder with deterministic attention (`ved_detAttn`)
3. Variational encoder-decoder with variational attention (`ved_varAttn`)

## Datasets
The proposed model and baselines have been evaluated on two experiments:
 1. Neural Question Generation
 with the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset
 2. Conversation Systems with the [Cornell Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset

The data has been preprocessed and the train-val-test split is provided in the `data/` directory.

## Requirements
- tensorflow-gpu==1.3.0
- Keras==2.0. 8
- numpy==1.12.1
- pandas==0.22.0
- gensim==3.1.2
- nltk==3.2.3
- tqdm==4.19.1

## Instructions
1. Generate word2vec, required for initializing word embeddings, specifying the dataset:
```
python w2v_generator.py --dataset qgen 
```
2. Train the desired model, set configurations in the `model_config.py` file. For example,
```
cd ved_varAttn
vim model_config.py # Make necessary edits
python train.py
``` 
- The model checkpoints are stored in `models/` directory, the summaries for Tensorboard are stored in `summary_logs/` directory. As training progresses, the metrics on the validation set are dumped into`log.txt`  and `bleu/` directory.
3. Evaluate performance of the trained model. Refer to `predict.ipynb` to load desired checkpoint, calculate performance metrics (BLEU and diversity score) on the test set, and generate sample outputs. 
