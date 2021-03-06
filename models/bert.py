import ktrain
from ktrain import text
import matplotlib.pyplot as plt
import tensorflow as tf

class BERT:
    def __init__(self):
        self.model = None
        self.learner = None
        self.predictor = None

    def fit(self, train_strings, y_train):
        tf.random.set_random_seed(0)
        (x_train, y_train), (x_test, y_test), preproc = \
            text.texts_from_array(train_strings, y_train, class_names=["low", "high"], preprocess_mode="bert", maxlen=300, lang="en")
        self.model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)
        learner = ktrain.get_learner(self.model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=12)
        self.learner = learner
        learner.fit_onecycle(1e-5,1)
        learner.plot('loss')
        plt.show()
        self.predictor = ktrain.get_predictor(learner.model, preproc)

    def find_lr(self):
        self.learner.lr_find(max_epochs=1, show_plot=True)
        plt.show()

    def predict(self, reviews_test):
        return [(0 if i == 'low' else 1) for i in self.predictor.predict(reviews_test)]

    def get_params(self, deep = True):
        return {'model':self.model, 'learner':self.learner, 'predictor':self.predictor}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
