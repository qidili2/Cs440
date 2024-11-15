"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict, Counter
def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag_stats = defaultdict(Counter)
    tag_frequency = Counter()
    
    for sentence in train:
        for word, tag in sentence:
            word_tag_stats[word][tag] += 1
            tag_frequency[tag] += 1 

    most_frequent_tag = {}
    for word, tag_count in word_tag_stats.items():
        most_frequent_tag[word] = tag_count.most_common(1)[0][0]
    
    most_frequent_overall_tag = tag_frequency.most_common(1)[0][0]
    
    predicted = []
    for sentence in test:
        tagged_sentence = []
        for word in sentence:
            if word in most_frequent_tag:
                tag = most_frequent_tag[word]
            else:
                tag = most_frequent_overall_tag 
            tagged_sentence.append((word, tag))
        predicted.append(tagged_sentence)
    
    return predicted