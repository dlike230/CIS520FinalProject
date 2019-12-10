import nltk
import numpy as np

from text_encoders.text_processing import samples_as_word_lists


class WordEncoder:

    def __init__(self):
        self.transform = None
        self.vocab_size = -1
        self.padded_length = -1

    def pad(self, encoding):
        if len(encoding) >= self.padded_length:
            return encoding[:self.padded_length]
        return encoding + [0] * (self.padded_length - len(encoding))

    def _transform(self, samples_as_word_lists, word_mapping):
        return np.array(
            [self.pad([word_mapping[word] if word in word_mapping else len(word_mapping) + 1 for word in word_list]) for
             word_list in samples_as_word_lists])

    def fit_transform(self, train_strings):
        words_per_sample = samples_as_word_lists(train_strings)
        self.padded_length = max(len(sample) for sample in words_per_sample)
        word_set = {word for sample in words_per_sample for word in sample}
        word_mapping = {}
        for i, word in enumerate(word_set):
            word_mapping[word] = i + 1
        self.vocab_size = len(word_set)
        self.transform = lambda texts: self._transform(samples_as_word_lists(texts), word_mapping)

        return self._transform(words_per_sample, word_mapping)
