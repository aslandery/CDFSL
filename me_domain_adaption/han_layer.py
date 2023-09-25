
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

conv1d = tf.layers.conv1d
flatten = tf.layers.flatten

def sp_attn_head(seq, bias_mat,out_sz, activation,conv_func=[], in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False, training=None):
    adj_mat = bias_mat

    seq = tf.cond(training, lambda : tf.nn.dropout(seq, 1.0 - in_drop), lambda : seq)
    seq_fts = conv_func[0](seq) 
    f_1 = conv_func[1](seq_fts)
    f_2 = conv_func[2](seq_fts)
    f_1 = tf.reshape(f_1, (adj_mat.shape[0], 1))
    f_2 = tf.reshape(f_2, (adj_mat.shape[0], 1))
    f_1 = adj_mat * f_1  
    f_2 = adj_mat * tf.transpose(f_2, [1, 0])  

    logits = tf.sparse_add(f_1, f_2)
    lrelu = tf.SparseTensor(indices=logits.indices,
                            values=tf.nn.leaky_relu(logits.values),
                            dense_shape=logits.dense_shape)
    coefs = tf.sparse.softmax(lrelu)

    coefs = tf.cond(training, lambda: tf.SparseTensor(indices=coefs.indices,
                                values=tf.nn.dropout(coefs.values, 1.0-coef_drop),
                                dense_shape=coefs.dense_shape),lambda:coefs)
  
    seq_fts = tf.cond(training, lambda: tf.nn.dropout(seq_fts, 1.0-in_drop),lambda: seq_fts)

    seq_fts = tf.squeeze(seq_fts) 
    vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
    vals = tf.expand_dims(vals, axis=0) 

    vals.set_shape([1, adj_mat.shape[0], out_sz]) 

    ret = vals + conv_func[3]

    if residual:
        if seq.shape[-1] != ret.shape[-1]:
            ret = ret + conv1d(seq, ret.shape[-1], 1)
        else:
            ret = ret + seq
    return activation(ret)


def attn_head(seq, out_sz, bias_mat, activation,conv_func=[], in_drop=0.0, coef_drop=0.0, residual=False,
              return_coef=False):
   
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        seq_fts = conv_func[0](seq)
        f_1 = conv_func[1](seq_fts)
        f_2 = conv_func[2](seq_fts)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1]) 
        with tf.control_dependencies([]):
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts) 
        ret = vals + conv_func[3]

        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) 
            else:
                seq_fts = ret + seq
        if return_coef:
            return activation(ret), coefs
        else:
            return activation(ret)  


def SimpleAttLayer(inputs, omega_weights=[], time_major=False, return_alphas=False,activation=tf.nn.relu):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])  


    w,b = omega_weights
    with tf.name_scope('v'):
    
        v1 = flatten(inputs)
        output = activation(tf.tensordot(v1, w, axes=1) + b )
    if not return_alphas:
        return output
    else:
        return output, alphas


def RSimpleAttLayer(inputs, omega_weights=[], time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])  

    w_omega, b_omega, u_omega = omega_weights
    with tf.name_scope('v'):
     
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')   
    alphas = tf.nn.softmax(vu, name='alphas')        
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas



class MeanAggregator(Layer):

    def __init__(self, units, input_dim, neigh_max, concat=True, dropout_rate=0.0, activation=tf.nn.relu, l2_reg=0,
                 use_bias=False,
                 seed=1024, type=None, **kwargs):
        super(MeanAggregator, self).__init__()
        self.units = units
        self.neigh_max = neigh_max
        self.concat = concat
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.input_dim = input_dim
        self.stype = type

    def build(self, input_shapes):
        with tf.device('/cpu:0'):
            self.neigh_weights = self.add_weight(shape=(self.input_dim, self.units),
                                                 initializer=glorot_uniform(
                                                     seed=self.seed),
                                                 regularizer=l2(self.l2_reg),
                                                 dtype=self.stype,
                                                 name="neigh_cancat_weight")
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.units), initializer=Zeros(), dtype=self.stype,
                                            name='neigh_cancat_bias')

            self.dropout = Dropout(self.dropout_rate)
            self.built = True

    def call(self, inputs, training=None):
        node_feat, neigh_feat = inputs
    
        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)

        concat_feat = tf.concat([neigh_feat, node_feat], axis=1)
        concat_mean = tf.reduce_mean(
            concat_feat, axis=1, keep_dims=False)   
        output = tf.matmul(concat_mean, self.neigh_weights)   
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        output._uses_learning_phase = True
        return output

    def get_config(self):
        config = {'units': self.units,
                  'concat': self.concat,
                  'seed': self.seed
                  }

        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolingAggregator(Layer):
    def __init__(self, units, input_dim, neigh_max, aggregator='meanpooling', concat=True,
                 dropout_rate=0.0,
                 activation=tf.nn.relu, l2_reg=0, use_bias=False,
                 seed=1024, stype=tf.float32):
        super(PoolingAggregator, self).__init__()
        self.output_dim = units
        self.input_dim = input_dim
        self.concat = concat
        self.pooling = aggregator
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.neigh_max = neigh_max
        self.seed = seed
        self.stype = stype

    def build(self, input_shapes):

        self.dense_layers = [Dense(
            self.input_dim, activation='relu', use_bias=True, kernel_regularizer=l2(self.l2_reg))]

        self.neigh_weights = self.add_weight(
            shape=(self.input_dim * 2, self.output_dim),
            initializer=glorot_uniform(
                seed=self.seed),
            regularizer=l2(self.l2_reg),
            dtype=self.stype,
            name="neigh_cancat_weight")

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=Zeros(),
                                        dtype=self.stype,
                                        name='neigh_cancat_bias')

        self.dropout = Dropout(self.dropout_rate)
        self.built = True

    def call(self, inputs, training=None):
        node_feat, neigh_feat = inputs

        node_feat = self.dropout(node_feat, training=training)
        neigh_feat = self.dropout(neigh_feat, training=training)

        dims = tf.shape(neigh_feat)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(
            neigh_feat, (batch_size * num_neighbors, self.input_dim))

        for l in self.dense_layers:
            h_reshaped = l(h_reshaped)
        neigh_feat = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))

        if self.pooling == "meanpooling":
            neigh_feat = tf.reduce_mean(neigh_feat, axis=1, keep_dims=False)
        else:
            neigh_feat = tf.reduce_max(neigh_feat, axis=1)   
        output = tf.concat(
            [tf.squeeze(node_feat, axis=1), neigh_feat], axis=-1)   

        output = tf.matmul(output, self.neigh_weights)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'concat': self.concat
                  }

        base_config = super(PoolingAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

