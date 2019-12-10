import tensorflow as tf

class MetricLoggingLayer(tf.keras.layers.Layer):

  def call(self, inputs):
    # The `aggregation` argument defines
    # how to aggregate the per-batch values
    # over each epoch:
    # in this case we simply average them.
    self.add_metric(tf.keras.backend.std(inputs),
                    name='std_of_activation',
                    aggregation='mean')
    return inputs