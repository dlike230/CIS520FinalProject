import ktrain
from ktrain import text
import matplotlib.pyplot as plt

class BERT:
    def __init__(self):
        self.model = None

    def fit(self, train_strings, y_train):
        (x_train, y_train), (x_test, y_test), preproc = \
            text.texts_from_array(train_strings, y_train, class_names=["low", "high"], preprocess_mode="bert", maxlen=400)
        self.model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)
        learner = ktrain.get_learner(self.model, train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=6)
        # learner.lr_find(max_epochs=1, show_plot=True)
        # plt.show()
        #learner.lr_plot()
        learner.fit_onecycle(2e-5, 2)
        self.predictor = ktrain.get_predictor(learner.model, preproc)

    def predict(self, reviews_test):
        return [(0 if i == 'low' else 1) for i in self.predictor.predict(reviews_test)]
