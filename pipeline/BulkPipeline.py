from pipeline.Pipeline import Pipeline


class BulkPipeline(Pipeline):

    def __init__(self, models, names):
        super().__init__("Score", 0.5)
        self.models = models
        self.names = names

    def make_model(self):
        return list(zip(self.names, self.models))

    def label_func(self, item):
        return 1 if item > 3 else 0
