# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import  Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0.2, pos_prior=0.5, silently=False):
    # Print out hyperparameter values
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    positive_unigram_counts = Counter()
    negative_unigram_counts = Counter()
    positive_bigram_counts = Counter()
    negative_bigram_counts = Counter()

    total_positive_unigrams = 0
    total_negative_unigrams = 0
    total_positive_bigrams = 0
    total_negative_bigrams = 0
    
    unigram_vocabulary = set()
    bigram_vocabulary = set()

    for doc, label in zip(train_set, train_labels):
        if label == 1:
            positive_unigram_counts.update(doc)
            total_positive_unigrams += len(doc)
            unigram_vocabulary.update(doc)
        else:
            negative_unigram_counts.update(doc)
            total_negative_unigrams += len(doc)
            unigram_vocabulary.update(doc)

        bigrams = list(zip(doc[:-1], doc[1:])) 
        if label == 1:
            positive_bigram_counts.update(bigrams)
            total_positive_bigrams += len(bigrams)
            bigram_vocabulary.update(bigrams)
        else:
            negative_bigram_counts.update(bigrams)
            total_negative_bigrams += len(bigrams)
            bigram_vocabulary.update(bigrams)

    vocab_size_unigram = len(unigram_vocabulary)
    vocab_size_bigram = len(bigram_vocabulary)

    yhats = []

    for doc in tqdm(dev_set, disable=silently):
        bigrams = list(zip(doc[:-1], doc[1:])) 
        log_prob_pos_unigram = math.log(pos_prior)
        log_prob_neg_unigram = math.log(1 - pos_prior)

        log_prob_pos_bigram = math.log(pos_prior)
        log_prob_neg_bigram = math.log(1 - pos_prior)

        for word in doc:
            word_prob_pos = (positive_unigram_counts[word] + unigram_laplace) / (total_positive_unigrams + unigram_laplace * vocab_size_unigram)
            log_prob_pos_unigram += math.log(word_prob_pos)

            word_prob_neg = (negative_unigram_counts[word] + unigram_laplace) / (total_negative_unigrams + unigram_laplace * vocab_size_unigram)
            log_prob_neg_unigram += math.log(word_prob_neg)

        for bigram in bigrams:
            bigram_prob_pos = (positive_bigram_counts[bigram] + bigram_laplace) / (total_positive_bigrams + bigram_laplace * vocab_size_bigram)
            log_prob_pos_bigram += math.log(bigram_prob_pos)

            bigram_prob_neg = (negative_bigram_counts[bigram] + bigram_laplace) / (total_negative_bigrams + bigram_laplace * vocab_size_bigram)
            log_prob_neg_bigram += math.log(bigram_prob_neg)

        final_log_prob_pos = (1 - bigram_lambda) * log_prob_pos_unigram + bigram_lambda * log_prob_pos_bigram
        final_log_prob_neg = (1 - bigram_lambda) * log_prob_neg_unigram + bigram_lambda * log_prob_neg_bigram

        if final_log_prob_pos > final_log_prob_neg:
            yhats.append(1) 
        else:
            yhats.append(0)  

    return yhats
