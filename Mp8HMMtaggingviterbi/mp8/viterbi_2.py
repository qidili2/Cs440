"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
Most of the code in this file is the same as that in viterbi_1.py
"""

from collections import defaultdict, Counter
from math import log

epsilon_for_pt = 1e-5
emit_epsilon = 1e-5

def training_v2(sentences):
    init_prob = defaultdict(lambda: 0)
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))
    alpha = 0.001

    tag_count = Counter()
    tag_pair_count = defaultdict(lambda: Counter())
    word_count = Counter()
    hapax_tag_count = Counter()

    total_sentences = len(sentences)

    for sentence in sentences:
        if sentence:
            first_tag = sentence[0][1]
            init_prob[first_tag] += 1

        for i, (word, tag) in enumerate(sentence):
            emit_prob[tag][word] += 1
            tag_count[tag] += 1
            word_count[word] += 1

            if i > 0:
                prev_tag = sentence[i - 1][1]
                tag_pair_count[prev_tag][tag] += 1

    hapax_words = {word for word, count in word_count.items() if count == 1}
    for sentence in sentences:
        for word, tag in sentence:
            if word in hapax_words:
                hapax_tag_count[tag] += 1

    total_hapax = sum(hapax_tag_count.values())
    hapax_prob = {
        tag: (count + 1) / (total_hapax + len(tag_count))
        for tag, count in hapax_tag_count.items()
    }

    total_tags = len(tag_count)
    for tag in init_prob:
        init_prob[tag] = (init_prob[tag] + alpha) / (total_sentences + alpha * total_tags)

    for tag, word_counts in emit_prob.items():
        V_T = len(word_counts)
        n_T = tag_count[tag]
        scaled_alpha = alpha * hapax_prob.get(tag, 1 / (total_hapax + len(tag_count)))

        for word in word_counts:
            emit_prob[tag][word] = (word_counts[word] + scaled_alpha) / (n_T + scaled_alpha * (V_T + 1))
        emit_prob[tag]["UNKNOWN"] = scaled_alpha / (n_T + scaled_alpha * (V_T + 1))

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

def viterbi_2(train, test, get_probs=training_v2):
    init_prob, emit_prob, trans_prob = get_probs(train)
    predicts = []

    for sentence in test:
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        for tag in init_prob:
            emit_p = emit_prob[tag].get(sentence[0], emit_prob[tag]["UNKNOWN"])
            init_p = init_prob.get(tag, epsilon_for_pt)
            prob_value = init_p * emit_p
            log_prob[tag] = log(prob_value) if prob_value > 0 else log(epsilon_for_pt)
            predict_tag_seq[tag] = [tag]

        for i in range(1, length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        final_probs = [(log_prob[tag], tag) for tag in log_prob if log_prob[tag] != float('-inf')]
        
        if final_probs:
            best_final_tag = max(final_probs, key=lambda x: x[0])[1]
            best_tag_seq = predict_tag_seq[best_final_tag]
        else:
            best_tag_seq = ["UNKNOWN"] * length
        
        predicts.append(list(zip(sentence, best_tag_seq)))
        
    return predicts
