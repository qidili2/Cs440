"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    alpha=0.001
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    tag_count = Counter()
    tag_pair_count = defaultdict(lambda: Counter())
    
    total_sentences = len(sentences)

    for sentence in sentences:
        if sentence:
            first_tag = sentence[0][1]
            init_prob[first_tag] += 1

        for i, (word, tag) in enumerate(sentence):
            emit_prob[tag][word] += 1
            tag_count[tag] += 1
            if i > 0:
                prev_tag = sentence[i - 1][1]
                tag_pair_count[prev_tag][tag] += 1

    # Normalize and apply Laplace smoothing
    total_tags = len(tag_count)
    for tag in init_prob:
        init_prob[tag] = (init_prob[tag] + alpha) / (total_sentences + alpha * total_tags)

    for tag, word_counts in emit_prob.items():
        V_T = len(word_counts)
        n_T = tag_count[tag]
        for word in word_counts:
            emit_prob[tag][word] = (word_counts[word] + alpha) / (n_T + alpha * (V_T + 1))
        emit_prob[tag]["UNKNOWN"] = alpha / (n_T + alpha * (V_T + 1))

    for prev_tag, next_tags in tag_pair_count.items():
        V_T = len(tag_count)
        n_T = sum(next_tags.values())
        for tag in next_tags:
            trans_prob[prev_tag][tag] = (next_tags[tag] + alpha) / (n_T + alpha * (V_T + 1))
        for tag in tag_count:
            if tag not in trans_prob[prev_tag]:
                trans_prob[prev_tag][tag] = alpha / (n_T + alpha * (V_T + 1))

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {}
    predict_tag_seq = {}

    if i == 0:
        for tag in emit_prob:
            emit_p = emit_prob[tag].get(word, emit_prob[tag]["UNKNOWN"])
            prob_value = prev_prob.get(tag, epsilon_for_pt) * emit_p
            log_prob[tag] = log(prob_value) if prob_value > 0 else log(epsilon_for_pt)
            predict_tag_seq[tag] = [tag]
    else:
        for curr_tag in emit_prob:
            max_prob, best_prev_tag = float('-inf'), None
            for prev_tag in prev_prob:
                trans_p = trans_prob[prev_tag].get(curr_tag, epsilon_for_pt)
                emit_p = emit_prob[curr_tag].get(word, emit_prob[curr_tag]["UNKNOWN"])
                if trans_p > 0 and emit_p > 0:
                    prob_value = prev_prob[prev_tag] + log(trans_p) + log(emit_p)
                    if prob_value > max_prob:
                        max_prob = prob_value
                        best_prev_tag = prev_tag

            if best_prev_tag is not None:
                log_prob[curr_tag] = max_prob
                predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_prev_tag] + [curr_tag]
            else:
                log_prob[curr_tag] = log(epsilon_for_pt)
                predict_tag_seq[curr_tag] = []

    return log_prob, predict_tag_seq


def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    predicts = []

    for sentence in test:
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}

        # Initialization
        for tag in init_prob:
            emit_p = emit_prob[tag].get(sentence[0], emit_prob[tag]["UNKNOWN"])
            init_p = init_prob.get(tag, epsilon_for_pt)
            prob_value = init_p * emit_p
            log_prob[tag] = log(prob_value) if prob_value > 0 else log(epsilon_for_pt)
            predict_tag_seq[tag] = [tag]

        # Forward step
        for i in range(1, length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        # Traceback
        final_probs = [(log_prob[tag], tag) for tag in log_prob if log_prob[tag] != float('-inf')]
        
        if final_probs:
            best_final_tag = max(final_probs, key=lambda x: x[0])[1]
            best_tag_seq = predict_tag_seq[best_final_tag]
        else:
            best_tag_seq = ["UNKNOWN"] * length
        
        predicts.append(list(zip(sentence, best_tag_seq)))
        
    return predicts