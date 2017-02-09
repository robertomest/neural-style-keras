'''
Custom Keras layers used on the pastiche model.
'''

import tensorflow as tf
import keras
from keras import initializations
from keras.layers import ZeroPadding2D, Layer, InputSpec

# Extending the ZeroPadding2D layer to do reflection padding instead.
class ReflectionPadding2D(ZeroPadding2D):
    def call(self, x, mask=None):
        pattern = [[0, 0],
                   [self.top_pad, self.bottom_pad],
                   [self.left_pad, self.right_pad],
                   [0, 0]]
        return tf.pad(x, pattern, mode='REFLECT')


class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, weights=None,
                 beta_init='zero', gamma_init='one', **kwargs):
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.epsilon = epsilon
        super(InstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # This currently only works for 4D inputs: assuming (B, H, W, C)
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (1, 1, 1, input_shape[-1])

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        self.built = True

    def call(self, x, mask=None):
        # Do not regularize batch axis
        reduction_axes = [1, 2]

        mean, var = tf.nn.moments(x, reduction_axes,
                                  shift=None, name=None, keep_dims=True)
        x_normed = tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, self.epsilon)
        return x_normed

    def get_config(self):
        config = {"epsilon": self.epsilon}
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConditionalInstanceNormalization(InstanceNormalization):
    def __init__(self, targets, nb_classes, **kwargs):
        self.targets = targets
        self.nb_classes = nb_classes
        super(ConditionalInstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # This currently only works for 4D inputs: assuming (B, H, W, C)
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (self.nb_classes, 1, 1, input_shape[-1])

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        self.built = True

    def call(self, x, mask=None):
        # Do not regularize batch axis
        reduction_axes = [1, 2]

        mean, var = tf.nn.moments(x, reduction_axes,
                                  shift=None, name=None, keep_dims=True)

        # Get the appropriate lines of gamma and beta
        beta = tf.gather(self.beta, self.targets)
        gamma = tf.gather(self.gamma, self.targets)
        x_normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, self.epsilon)

        return x_normed
