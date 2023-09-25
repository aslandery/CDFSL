import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from normalization import forward_modulate


class RgcnAggregator(Layer):
    def __init__(self, units, input_dim,
                 dropout_rate=0.0,
                 activation=tf.nn.relu, l2_reg=0, use_bias=False,
                 seed=1024, mp_num=0):
        super(RgcnAggregator, self).__init__()
        self.output_dim = units
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.activation = activation
        self.seed = seed
        self.relation = mp_num

    def build(self, input_shapes):
        self.relation_weights = []
        self.weight = self.add_weight(
            shape=(self.output_dim * (self.relation+1), self.output_dim),
            initializer=glorot_uniform(
               ),
            regularizer=l2(self.l2_reg),
            name="concat_weight")

        for i in range(self.relation):
            matw = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=glorot_uniform(),
                regularizer=l2(self.l2_reg),
                name="relation" + str(i) + "_weights")
            self.relation_weights.append(matw)

        self.self_weights = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=glorot_uniform(
               ),
            regularizer=l2(self.l2_reg),
            name="self_weight")

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=Zeros(),
                                        name='bias_weight')
        self.built = True

    def call(self, inputs, mask=None):
        node_feat, relation_feat = inputs
        batch_size = tf.shape(node_feat)[0]

        with tf.control_dependencies([]):
            h_reshaped = tf.reshape(
                relation_feat, [batch_size, self.relation, -1, self.input_dim])
            relation_output = []
            for r in range(self.relation):
                rel_reshaped = tf.gather(h_reshaped, r, axis=1)  
                rel_reshaped = tf.reshape(rel_reshaped, [batch_size, -1, self.input_dim])  
                rel_reshaped = tf.reduce_mean(rel_reshaped, axis=1)
                rel_feat = tf.matmul(rel_reshaped, self.relation_weights[r])
                relation_output.append(rel_feat) 

        node_feat = tf.reshape(node_feat, [-1, self.input_dim])
        node_feat = tf.matmul(node_feat, self.self_weights)
        relation_output.append(node_feat) 
        relation_emb = tf.concat(relation_output, axis=-1)  
        output = tf.matmul(relation_emb, self.weight)
        if self.use_bias:
            output += self.bias
        if self.activation:
            output = self.activation(output)
        return output


def set_rgcn_layers(n_classes, n_hidden, input_feature_dim, neighbor_num, activation, l2_reg,
            use_bias, dropout_rate, mp_num):
    aggregator = RgcnAggregator
    agg_model = list()
    for l in range(0, len(neighbor_num)):  
        if l == 0:
            feature_dim = input_feature_dim
        if l > 0:
            feature_dim = n_hidden
        if l == len(neighbor_num) - 1:
            n_hidden = n_classes

        agg = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation,
                         l2_reg=l2_reg, use_bias=use_bias, dropout_rate=dropout_rate, mp_num=mp_num)
        agg_model.append(agg)
    return agg_model

def set_rgcn_layers(n_classes, n_hidden, input_feature_dim, neighbor_num, activation, l2_reg,
            use_bias, dropout_rate, mp_num):
    aggregator = RgcnAggregator
    agg_model = list()
    for l in range(0, len(neighbor_num)):  
        if l == 0:
            feature_dim = input_feature_dim
        if l > 0:
            feature_dim = n_hidden
        if l == len(neighbor_num) - 1:
            n_hidden = n_classes

        agg = aggregator(units=n_hidden, input_dim=feature_dim, activation=activation,
                         l2_reg=l2_reg, use_bias=use_bias, dropout_rate=dropout_rate, mp_num=mp_num)
        agg_model.append(agg)
    return agg_model

def GraphSAGE_rgcn(input_list, feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu,
                    dropout_rate=0.0, l2_reg=0, training=None, mp_num=0, train_ind_tensor=None):

    features = input_list[0]
    node_input = input_list[1]
    hop1s = input_list[2]
    hop2s = input_list[3]
    if train_ind_tensor is None:
        toy_ind = tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.int32)

    ori_h = features
    if train_ind_tensor is None:
        node_input_cur = tf.gather(node_input, toy_ind)
        hop1 = tf.gather(hop1s, toy_ind)  
        hop2 = tf.gather(hop2s, toy_ind)
    else:
        node_input_cur = tf.gather(node_input, train_ind_tensor)
        hop1 = tf.identity(tf.gather(hop1s, train_ind_tensor))
        hop2 = tf.gather(hop2s, train_ind_tensor)
    model = set_rgcn_layers(n_classes, n_hidden, feature_dim, neighbor_num, activation, l2_reg,
                        use_bias, dropout_rate, mp_num)

    hops = [node_input_cur, hop1, hop2]
    ori_batch = tf.shape(node_input_cur)[0]
    for l in range(0, len(neighbor_num)):  
        after1agg = []
        if l == 0:
            input_dim = feature_dim
        if l > 0:
            input_dim = n_hidden
        for hop in range(len(neighbor_num) - l):  
            if l == 0:

                node_input_feat = tf.gather(ori_h, hops[hop]) 

                node_input_feat = tf.reshape(
                    node_input_feat, [-1, 1, input_dim]) 


                hop1_feat = tf.gather(ori_h, hops[hop + 1]) 

                hop1_feat = tf.reshape(
                    hop1_feat, [-1, mp_num * neighbor_num[hop], input_dim])

            else:
                
                node_input_feat = mid_h[hop] 
                
                node_input_feat = tf.reshape(
                    node_input_feat, [-1, 1, input_dim])  

                hop1_feat = mid_h[hop + 1]

                hop1_feat = tf.reshape(
                    hop1_feat, [-1, neighbor_num[hop] * mp_num, input_dim])  

            h = model[l]([node_input_feat, hop1_feat], training)
          
            h = tf.reshape(h, [ori_batch, -1, n_hidden])
            after1agg.append(h)
        mid_h = after1agg  
    after2hop = mid_h[0]

    emb = tf.reshape(after2hop, (-1, n_hidden))
    with tf.control_dependencies([]):
        return tf.identity(emb)


def forward_rgcn_layer(layer_input, ind, weights, dropout_rate=0.3, activation=tf.nn.relu,
                                             train_flag=False, neighbor_num=[10,25], mp_num=2, seed=0):
    def rgcnagg(feat, rgcnweights):
        node_feat, relation_feat = feat
        node_feat = Dropout(dropout_rate)(node_feat, training=train_flag)
        relation_feat = Dropout(dropout_rate)(relation_feat, training=train_flag)
        cancat_weights = rgcnweights[0]
        agg_weights = rgcnweights[1:]
        agg_dim = tf.shape(node_feat)[-1]
        agg_batch = tf.shape(node_feat)[0]
        with tf.control_dependencies([]):
  
            h_reshaped = tf.reshape(
                relation_feat, [agg_batch, mp_num, -1, input_dim])
           
            relation_output = []
            for r in range(mp_num):
                rel_reshaped = tf.gather(h_reshaped, r, axis=1)   
                rel_reshaped = tf.reshape(rel_reshaped, [agg_batch,-1, agg_dim])   
                rel_reshaped = tf.reduce_mean(rel_reshaped, axis=1)
                rel_feat = tf.matmul(rel_reshaped, agg_weights[r])
                relation_output.append(rel_feat) 
        node_feat = tf.reshape(node_feat, [-1, input_dim])
        node_feat = tf.matmul(node_feat, agg_weights[mp_num])
        relation_output.append(node_feat) 
        relation_emb = tf.concat(relation_output, axis=-1) 
        output = tf.matmul(relation_emb, cancat_weights)
        output += agg_weights[mp_num+1]
        output = activation(output)
        return output

    features, relation_node, hops1, hops2 = layer_input
    node_input = tf.gather(relation_node, ind)
    hop1 = tf.gather(hops1, ind)
    hop2 = tf.gather(hops2, ind)

    feature_dim = tf.shape(features)[-1]
    n_hidden = 32
    ori_batch = tf.shape(node_input)[0]
    ori_h = features
    if mp_num == 1:
        rgcnl= 4
    elif mp_num ==2:    
        rgcnl = 5
    elif mp_num == 3:
        rgcnl = 6
    hops = [node_input, hop1, hop2]
    for l in range(0, len(neighbor_num)):  
        after1agg = []
        if l == 0:
            input_dim = feature_dim
        if l > 0:
            input_dim = n_hidden
        for hop in range(len(neighbor_num) - l):  
            if l == 0:
                node_input_feat = tf.gather(ori_h, hops[hop])  

                node_input_feat = tf.reshape(
                    node_input_feat, [-1, 1, input_dim])  

                hop1_feat = tf.gather(ori_h, hops[hop + 1])  

                hop1_feat = tf.reshape(
                    hop1_feat, [-1, mp_num * neighbor_num[hop], input_dim])

            else:
                node_input_feat = mid_h[hop]  
                node_input_feat = tf.reshape(
                    node_input_feat, [-1, 1, input_dim])   

                hop1_feat = mid_h[hop + 1]

                hop1_feat = tf.reshape(
                    hop1_feat, [-1, neighbor_num[hop] * mp_num, input_dim])  

            h = rgcnagg([node_input_feat, hop1_feat], weights[l *
                                                              rgcnl:(l + 1) * rgcnl])

            h = tf.reshape(h, [ori_batch, -1, n_hidden])   
            after1agg.append(h)
        mid_h = after1agg  
    after2hop = mid_h[0]
    emb = tf.reshape(after2hop, (-1, n_hidden))
    return emb


def forward_rgcn(input_list, ind, weights, training_flag=None, reuse=False, mp_num=2, note='',
                     neighbor_num=[10, 25], sa='', ss='', dropout=0.0, modulate=False):

    with tf.variable_scope('forward_rgcn', reuse=reuse):
        n_heads = [1, 1]
        hid_units = [32]
        activation = tf.nn.elu
        ffd_drop = 0.5
        attn_drop = 0.5
        if 'ss' in ss:
            num_sa_ = 4
        else:
            num_sa_ = 0
        if mp_num == 2:
            num_agg = 5
        elif mp_num==3:
            num_agg = 6
        elif mp_num ==1:
            num_agg = 4
        dense_k = weights[-2]
        dense_b = weights[-1]

        ft = input_list[0]
        no = input_list[1]
        hop1 = input_list[2]
        hop2 = input_list[3]
        bc_weights = weights[num_sa_:]
        modulate_weights = bc_weights[num_sa_+num_agg*len(neighbor_num):-2]
        with tf.control_dependencies([]):
            nb_nodes = tf.shape(ft)[0]
            final_embed = forward_rgcn_layer(input_list, ind, bc_weights, dropout_rate=dropout, activation=activation,
                                         train_flag=training_flag, neighbor_num=neighbor_num, mp_num=mp_num)

            modulate = False
            if modulate:
                modulate_mask = forward_modulate(final_embed, modulate_weights, activation, dropout, training_flag)
                final_embed = final_embed * modulate_mask
                logits = tf.tensordot(final_embed, dense_k, axes=1) + dense_b
            else:
                with tf.control_dependencies([]):
                    logits = tf.tensordot(final_embed, dense_k, axes=1) + dense_b

            return logits,final_embed,[]
