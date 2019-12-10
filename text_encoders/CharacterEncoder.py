import nltk
import numpy as np

from text_encoders.text_processing import samples_as_word_lists


class CharacterRNNEncoder:

    def __init__(self):
        self.transform = None
        self.vocab_size = -1
        self.padded_length = -1

    def pad(self, encoding):
        if len(encoding) >= self.padded_length:
            return encoding[:self.padded_length]
        return encoding + [0] * (self.padded_length - len(encoding))

    def _transform(self, samples, character_mapping):
        return np.array(
            [self.pad(
                [character_mapping[character] if character in character_mapping else len(character_mapping) + 1 for
                 character in
                 characters]) for
             characters in samples])

    def fit_transform(self, train_strings):
        self.padded_length = max(len(sample) for sample in train_strings)
        character_set = {c for sample in train_strings for c in sample}
        character_mapping = {}
        for i, character in enumerate(character_set):
            character_mapping[character] = i + 1
        self.vocab_size = len(character_set)
        self.transform = lambda texts: self._transform(samples_as_word_lists(texts), character_mapping)

        return self._transform(train_strings, character_mapping)
