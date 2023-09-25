import tensorflow as tf
from tensorflow.python.keras.initializers import Identity, glorot_uniform, Zeros
from tensorflow.python.keras.layers import Dropout, Input, Layer, Embedding, Reshape, Dense
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model


class OneLayerClassifier(Layer):   
    def __init__(self, units,
                 activation=tf.nn.softmax, dropout_rate=0.5,
                 use_bias=True, l2_reg=0,
                 seed=1024, **kwargs):
        super(OneLayerClassifier, self).__init__(**kwargs)
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

        self.kernel = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=glorot_uniform(
                                          seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      dtype= self.stype,
                                      name='class_kernel', )
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        dtype= self.stype,
                                        name='class_bias', )
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        features = self.dropout(inputs, training=training)  
        output = tf.matmul(features, self.kernel)
        if self.bias is not None:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        return output


class SsEncoder(Layer):   
    def __init__(self, units,
                 activation=tf.nn.relu, dropout_rate=0.5,
                 use_bias=True, l2_reg=0,
                 seed=1024, **kwargs):
        super(SsEncoder, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.seed = seed
        self.stype = tf.float32

    def build(self, input_shapes): 
        input_dim = int(input_shapes[-1])

        self.kernel = self.add_weight(shape=(input_dim,
                                             self.units),
                                      initializer=glorot_uniform(
                                          seed=self.seed),
                                      regularizer=l2(self.l2_reg),
                                      dtype= self.stype,
                                      name='ss_ontop_kernel', )
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=Zeros(),
                                        dtype=self.stype,
                                        name='ss_ontop_bias', )
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        features = self.dropout(inputs, training=training)   
        output = tf.matmul(features, self.kernel)
        if self.bias is not None:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        return output


class MLP_FS(tf.keras.Model):
    def __init__(self, units, activation, dropout_rate, use_bias=True, l2_reg=0,  seed=1024):
        super(MLP_FS, self).__init__(self)
        self.seed = seed
        self.use_bias = use_bias
        self.l2_reg = l2_reg
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)

        self.dense1 = tf.keras.layers.Dense(units=self.units[0], activation=self.activation[0], use_bias=self.use_bias,
                                            kernel_initializer=glorot_uniform(seed=self.seed),
                                            kernel_regularizer=l2(self.l2_reg))
        self.dense2 = tf.keras.layers.Dense(units=self.units[1], activation=self.activation[1], use_bias=self.use_bias,
                                            kernel_initializer=glorot_uniform(seed=self.seed),
                                            kernel_regularizer=l2(self.l2_reg))

    def call(self, inputs, training=None, **kwargs):  
        features, ind = inputs

        first_input = self.dropout(features, training=training)
        first_output = self.dense1(first_input)
        second_input = self.dropout(first_output, training=training)
        x = self.dense2(second_input)
        x = tf.gather(x, ind)
        return x
