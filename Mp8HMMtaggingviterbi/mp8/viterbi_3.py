"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
from collections import defaultdict, Counter
from math import log

# Constants
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5

def training(sentences):
    alpha = 0.1
    init_prob, emit_prob, trans_prob = defaultdict(float), defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
    emit_count, trans_count, tag_count = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(int)
    word_count = Counter()
    count_occurrences(sentences, emit_count, trans_count, tag_count, word_count)
    init_prob = compute_initial_prob(trans_count, len(sentences))
    hapax_prob = compute_hapax_probabilities(emit_count, word_count)
    emit_prob = compute_emission_probabilities(emit_count, tag_count, hapax_prob, alpha)
    trans_prob = compute_transition_probabilities(trans_count, alpha)

    return init_prob, emit_prob, trans_prob


def count_occurrences(sentences, emit_count, trans_count, tag_count, word_count):
    for sentence in sentences:
        prev_tag = "START"
        for word, tag in sentence:
            emit_count[tag][word] += 1
            trans_count[prev_tag][tag] += 1
            tag_count[tag] += 1
            word_count[word] += 1
            prev_tag = tag
        trans_count[prev_tag]["END"] += 1


def compute_initial_prob(trans_count, total_sentences):
    return {tag: count / total_sentences for tag, count in trans_count["START"].items()}


def compute_hapax_probabilities(emit_count, word_count):
    hapax_words = {word for word, count in word_count.items() if count == 1}
    hapax_tag_type_count = defaultdict(lambda: defaultdict(int))
    total_hapax_count = 0

    for tag, words in emit_count.items():
        for word in words:
            if word in hapax_words:
                word_type = classify_word(word)
                hapax_tag_type_count[tag][word_type] += 1
                total_hapax_count += 1

    hapax_prob = defaultdict(lambda: defaultdict(float))
    for tag in emit_count:
        for word_type in ["digit", "very short", "short end s", "short", "long end s", "long"]:
            hapax_tag_type_count[tag][word_type] = hapax_tag_type_count[tag].get(word_type, 1)
            hapax_prob[tag][word_type] = hapax_tag_type_count[tag][word_type] / total_hapax_count

    return hapax_prob


def compute_emission_probabilities(emit_count, tag_count, hapax_prob, alpha):
    emit_prob = defaultdict(lambda: defaultdict(float))

    for tag, word_dict in emit_count.items():
        total_tag_count = sum(word_dict.values())
        vocab_size = len(word_dict)
        alpha_e_scaled_total = sum(alpha * hapax_prob[tag][word_type] for word_type in hapax_prob[tag])

        for word, count in word_dict.items():
            P_e = (count + alpha_e_scaled_total) / (total_tag_count + alpha_e_scaled_total * (vocab_size + 1))
            emit_prob[tag][word] = P_e

        for word_type in ["digit", "very short", "short end s", "short", "long end s", "long"]:
            alpha_e_scaled = alpha * hapax_prob[tag][word_type]
            P_e_unknown = alpha_e_scaled / (total_tag_count + alpha_e_scaled * (vocab_size + 1))
            emit_prob[tag][word_type] = P_e_unknown

    return emit_prob


def compute_transition_probabilities(trans_count, alpha):
    trans_prob = defaultdict(lambda: defaultdict(float))

    for tag0, tag_counts in trans_count.items():
        total_tag0_count = sum(tag_counts.values())
        vocab_size = len(tag_counts)

        for tag1, count in tag_counts.items():
            P_t = (count + alpha) / (total_tag0_count + alpha * (vocab_size + 1))
            trans_prob[tag0][tag1] = P_t

    return trans_prob


def classify_word(word):
    if word[0].isdigit() and word[-1].isdigit():
        return "digit"
    elif len(word) <= 3:
        return "very short"
    elif len(word) >= 10:
        return "long end s" if word.endswith('s') else "long"
    elif 4 <= len(word) <= 9:
        return "short end s" if word.endswith('s') else "short"
def viterbi_3(train, test):
    init_prob, emit_prob, trans_prob = training(train)
    predicts = []

    for sentence in test:
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}

        word_type = classify_word(sentence[0])
        for tag in init_prob:
            emit_p = emit_prob[tag].get(sentence[0], emit_prob[tag].get(word_type, emit_epsilon))
            init_p = init_prob.get(tag, epsilon_for_pt)
            prob_value = init_p * emit_p
            log_prob[tag] = log(prob_value) if prob_value > 0 else log(epsilon_for_pt)
            predict_tag_seq[tag] = [tag]

        for i in range(1, length):
            word_type = classify_word(sentence[i])
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        final_probs = [(log_prob[tag], tag) for tag in log_prob if log_prob[tag] != float('-inf')]

        if final_probs:
            best_final_tag = max(final_probs, key=lambda x: x[0])[1]
            best_tag_seq = predict_tag_seq[best_final_tag]

        predicts.append(list(zip(sentence, best_tag_seq)))

    return predicts




def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    log_prob = {}
    predict_tag_seq = {}
    word_type = classify_word(word)

    if i == 0:
        for tag in emit_prob:
            emit_p = emit_prob[tag].get(word, emit_prob[tag].get(word_type, emit_epsilon))
            prob_value = prev_prob.get(tag, epsilon_for_pt) * emit_p
            log_prob[tag] = log(prob_value) if prob_value > 0 else log(epsilon_for_pt)
            predict_tag_seq[tag] = [tag]
    else:
        for curr_tag in emit_prob:
            max_prob, best_prev_tag = float('-inf'), None
            for prev_tag in prev_prob:
                trans_p = trans_prob[prev_tag].get(curr_tag, epsilon_for_pt)
                emit_p = emit_prob[curr_tag].get(word, emit_prob[curr_tag].get(word_type, emit_epsilon))
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