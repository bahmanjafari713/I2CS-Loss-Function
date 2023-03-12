import tensorflow as tf
class I2CS_layer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(I2CS_layer, self).__init__()
        self.units = units
    def build(self, input_shape):
        self.c = self.add_weight(
            shape=(self.units, input_shape[-1]),
            initializer="random_normal",
            trainable=True,
        )
    def call(self, inputs):
        margin=20
        d_intra=tf.square(tf.transpose([tf.reduce_sum(tf.square(inputs), axis=-1)]) + \
                             tf.reduce_sum(tf.square(self.c), axis=-1) - \
                             2 * tf.matmul(inputs, tf.transpose(self.c)))
    
        d_inter = tf.reduce_min(tf.eye(self.units, self.units) * 1e9+
           tf.transpose([tf.reduce_sum(tf.square(self.c), axis=-1)]) +
            tf.reduce_sum(tf.square(self.c), axis=-1) -
            2 * tf.matmul(self.c, tf.transpose(self.c))
            , axis=-1)
        return d_intra+margin/d_inter


def I2CS_loss(y_true, y_pred):
    return tf.reduce_sum(tf.multiply(tf.cast(y_true, tf.float32), y_pred))

def I2CS_accuracy(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, -y_pred)