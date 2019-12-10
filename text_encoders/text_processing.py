import nltk


def samples_as_word_lists(samples):
    return [[word for sentence in nltk.sent_tokenize(train_string) for word in
             nltk.word_tokenize(sentence)] for train_string in samples]