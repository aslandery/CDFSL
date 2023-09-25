from tensorflow.python.keras import backend as K
from tensorflow.contrib.layers import layer_norm
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Conv2D

import tensorflow as tf
import tensorflow.nn as nn
from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.python.keras.regularizers import l2

from tensorflow.python.keras.initializers import Identity, glorot_uniform, Zeros

def batch_norm_layer(x, train_phase, scope_bn):
  bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
                        updates_collections=None,
                        is_training=True,
                        reuse=None, 
                        trainable=True,
                        scope=scope_bn)
  bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
                            updates_collections=None,
                            is_training=False,
                            reuse=True,  
                            trainable=True,
                            scope=scope_bn)
  z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
  return z


class Modulate(Layer):  
    def __init__(self, units,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0,
                 seed=1024, **kwargs):
        super(Modulate, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.seed = seed
        self.stype = tf.float32

    def build(self, input_shapes): 

        assert len(input_shapes) == 2
        input_dim = int(input_shapes[-1])

        self.layer1w = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=glorot_uniform(
                                          seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      dtype= self.stype,
                                      name='modulate1w', )

        if self.use_bias:
            self.layer1b = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        dtype= self.stype,
                                        name='modulate1b', )


        self.layer2w = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=glorot_uniform(
                                          seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      dtype= self.stype,
                                      name='modulate2w', )
        if self.use_bias:
            self.layer2b = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        dtype= self.stype,
                                        name='modulate2b', )

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.built = True

    def call(self, inputs, training=None):
       
        return (inputs * self.layer1w + self.layer1b ) * self.layer2w + self.layer2b

def forward_modulate_ori(inputs, weights,act, drop, training):
    if training is True:
        inputs = drop(inputs)
    w1, b1, w2, b2 = weights
    h1 = inputs * w1 + b1
    h1 = act(h1)
    h2 = h1 * w2 + b2
    h2 = act(h2)
    code = tf.reduce_mean(h2, axis=0)
    return code

def forward_modulate(inputs, weights,act, drop, training):
    if training is True:
        inputs = drop(inputs)
    w1, b1, w2, b2 = weights
    h1 = tf.matmul(inputs , w1 )+ b1
    h1 = act(h1)
    h2 = tf.matmul(h1 , w2) + b2
    h2 = act(h2)
    code = tf.reduce_mean(h2, axis=0)
    return code


class Reg(Layer):  
    def __init__(self, units,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0,
                 seed=1024, **kwargs):
        super(Reg, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.seed = seed
        self.stype = tf.float32

    def build(self, input_shapes):  

        assert len(input_shapes) == 2
        input_dim = int(input_shapes[-1])

        self.layer1w = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=glorot_uniform(
                                          seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      dtype= self.stype,
                                      name='regloss1w', )

        if self.use_bias:
            self.layer1b = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        dtype= self.stype,
                                        name='regloss1b', )


        self.layer2w = self.add_weight(shape=(self.units,
                                             16),
                                      initializer=glorot_uniform(
                                          seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      dtype= self.stype,
                                      name='regloss2w', )
        if self.use_bias:
            self.layer2b = self.add_weight(shape=(16,),
                                        initializer=Zeros(),
                                        dtype= self.stype,
                                        name='regloss2b', )

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.built = True

    def call(self, inputs, training=None):

        return tf.matmul(inputs * self.layer1w + self.layer1b , self.layer2w) + self.layer2b

def forward_reg(inputs, weights, act=tf.nn.relu, drop=0, training=False):
    if len(weights) > 4:
        weights = weights[-6:-2]
    if training is True:
        inputs = Dropout(drop)(inputs)
    w1, b1, w2, b2 = weights
    h1 = tf.matmul(inputs , w1 )+ b1
    h1 = act(h1)
    h2 = tf.matmul(h1 , w2) + b2
    h2 = act(h2)
    code = tf.reduce_mean(h2, axis=0)
    return code


class FeatureWiseTransform(Layer):
  def __init__(self, units, w_init, b_init):
    super(FeatureWiseTransform, self).__init__()
    self.units = units
    self.w_init = w_init
    self.b_init = b_init

  def build(self, input_shapes):
    dim = self.units
    self.gamma = self.add_weight(shape=(1, dim), initializer=tf.constant_initializer(  self.w_init), name='fwt_gamma' )
    self.beta = self.add_weight(shape=(1, dim), initializer=tf.constant_initializer(  self.b_init), name='fwt_beta')
    self.built = True

  def call(self, inputs, training=None):
    dim = self.gamma.shape[-1]
    if training:
      gamma = 1 + tf.random_normal((1, dim), dtype=self.gamma.dtype) * tf.math.softplus(self.gamma)
      beta = tf.random_normal((1, dim), dtype=self.beta.dtype) * tf.math.softplus(self.beta)
      output = gamma * inputs + beta

    else:
      output = inputs
    return output

def fwt_forward(inputs, fwt_weights):
  gamma, beta = fwt_weights
  output = gamma * inputs + beta
  return output
