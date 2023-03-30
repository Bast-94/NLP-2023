# Introduction to NLP 01
# Lab 02 by François Soulier, Raphael Bennaim and Bastien Hoorelbeke
## Short description of the project
This project involves coding a sentiment classifier on the IMDB sentiment dataset. It is divided in three main parts:
* Pre-processing data: getting the data and remove unuseful features in order to work on clean data.
* Naive Bayes Classifier: Implementation of a classifier based on conditional probabilities and another one which is using sickit-learn.
* Stemming and Lemmatization: Adding stemming or lemmatization to the pretreatment.
## Description of the file/module architecture
 * [Lab02.ipynb](./Lab02.ipynb): Main notebook
 * [scripts](./scripts)
   * [data.py](./scripts/data.py): Pretreatment utils
   * [naive_bayes](./scripts/naive_bayes)
      * [from_scratch.py](./scripts/naive_bayes/from_scratch.py): Naive Bayes classifier with probabilities
      * [scikit_learn.py](./scripts/naive_bayes/scikit_learn.py): Naive Bayes classifier with scikit learn
 * [README.md](./README.md)
