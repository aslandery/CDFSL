import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Input, Dense, Dropout, Layer, LSTM
from han_layer import MeanAggregator, PoolingAggregator, SimpleAttLayer, RSimpleAttLayer
from normalization import FeatureWiseTransform


def set_agg_fwt(n_classes, n_hidden, input_feature_dim, neighbor_num, activation, l2_reg,
            use_bias, dropout_rate, aggregator_type, stype, fwt, w_init, b_init):
    with tf.variable_scope('set_agg_fwt', reuse=tf.AUTO_REUSE):
        agg_model = []
        fwt_model = []
        if aggregator_type == 'mean':
            aggregator = MeanAggregator
        elif aggregator_type == 'pooling':
            aggregator = PoolingAggregator
        elif aggregator_type == 'att_g':
            aggregator = AttMeanAggregator

        if fwt:
            fwt_layer = FeatureWiseTransform

        if fwt:
            fwt_model = fwt_layer(units=input_feature_dim, w_init=w_init, b_init=b_init)

        for l in range(0, len(neighbor_num)):  
            if l == 0:
                feature_dim = input_feature_dim
            if l > 0:
                feature_dim = n_hidden
            if l == len(neighbor_num) - 1:
                n_hidden = n_classes

            agg = aggregator(units=n_hidden, input_dim=feature_dim, neigh_max=neighbor_num[l], activation=activation,
                             l2_reg=l2_reg, use_bias=use_bias, dropout_rate=dropout_rate, aggregator=aggregator_type,
                             stype=stype)
            agg_model.append(agg)
        if fwt:
            return agg_model, fwt_model
        else:
            return agg_model


def GraphSAGE_fwt(input_list, input_feature_dim, neighbor_num, n_hidden, n_classes, use_bias=True, activation=tf.nn.relu,
                 aggregator_type='mean', dropout_rate=0.0, l2_reg=0, training=None, mp_num=1, stype=None, sa='',
                 sa_size=64,
                 train_ind_tensor=None, fwt=False, w_init=1, b_init=0):

    if sa == 'mlp':
        mp_attr = n_hidden * mp_num
        w_omega = tf.get_variable(
            'semantic_w_o', (mp_attr, sa_size), initializer=glorot_uniform(), dtype=stype)
        b_omega = tf.get_variable(
            'semantic_bias_o', (sa_size), initializer=Zeros(), dtype=stype)
        att_slayer = [w_omega, b_omega]
    elif sa == 'sa':
        hidden_dim = n_hidden
        w_omega = tf.get_variable(
            'semantic_w_o', (hidden_dim, sa_size), initializer=glorot_uniform(), dtype=stype)
        b_omega = tf.get_variable(
            'semantic_bias_o', (sa_size), initializer=Zeros(), dtype=stype)
        u_omega = tf.get_variable(
            'semantic_u_o', (sa_size), initializer=glorot_uniform(), dtype=stype)
        att_slayer = [w_omega, b_omega, u_omega]

    n_classes = n_hidden
    if fwt:
        model, fwt_model = set_agg_fwt(n_classes, n_hidden, input_feature_dim, neighbor_num, activation, l2_reg,
                    use_bias, dropout_rate, aggregator_type, stype, fwt, w_init, b_init)
    else:
        model = set_agg_fwt(n_classes, n_hidden, input_feature_dim, neighbor_num, activation, l2_reg,
                            use_bias, dropout_rate, aggregator_type, stype, fwt, w_init, b_init)
    features = tf.random.uniform(shape=[10,128])
    node_input = tf.constant(range(10), dtype=tf.int32)
    hop1s = [tf.ones(shape=[10,10],dtype=tf.int32) for i in range(mp_num)]
    hop2s = [tf.ones(shape=[10,25*10],dtype=tf.int32) for i in range(mp_num)]


    embed_list = []
    if train_ind_tensor is None:
        toy_ind = tf.random.uniform(shape=(1,), minval=0, maxval=1, dtype=tf.int32)

    for hop1, hop2 in zip(hop1s, hop2s):
        if fwt:
            ori_h = fwt_model(features)
        else:
            ori_h = features
        if train_ind_tensor is None:
            node_input_cur = tf.gather(node_input, toy_ind)
            hop1 = tf.gather(hop1, toy_ind)
            hop2 = tf.gather(hop2, toy_ind)
        else:
            node_input_cur = tf.gather(node_input, train_ind_tensor)

            hop1 = tf.identity(tf.gather(hop1, train_ind_tensor))
            hop2 = tf.gather(hop2, train_ind_tensor)
        hops = [node_input_cur, hop1, hop2]
        for l in range(0, len(neighbor_num)):  
            after1agg = []
            if l == 0:
                feature_dim = input_feature_dim
            if l > 0:
                feature_dim = n_hidden
            for hop in range(len(neighbor_num) - l):  
                if l == 0:
                    node_input_feat = tf.gather(ori_h, hops[hop])
                    node_input_feat = tf.reshape(
                        node_input_feat, [-1, 1, feature_dim])
                    hop1_feat = tf.gather(ori_h, hops[hop + 1])
                    hop1_feat = tf.reshape(
                        hop1_feat, [-1, neighbor_num[hop], feature_dim])
                else:
                    node_input_feat = mid_h[hop] 
                    node_input_feat = tf.reshape(
                        node_input_feat, [-1, 1, feature_dim])
                    hop1_feat = mid_h[hop + 1]
                    hop1_feat = tf.reshape(
                        hop1_feat, [-1, neighbor_num[hop], feature_dim]) 

                h = model[l]([node_input_feat, hop1_feat], training)
                after1agg.append(h)
            mid_h = after1agg  
        after2hop = mid_h[0]
        embed_list.append(tf.expand_dims(tf.reshape(
            after2hop, (-1, n_hidden)), axis=1))  

    multi_embed = tf.concat(embed_list, axis=1)
    if sa == 'mlp':
        with tf.control_dependencies([]):
            final_embed = SimpleAttLayer(multi_embed, att_slayer,
                                         time_major=False,
                                         return_alphas=False)
    elif sa == 'sa':
        final_embed = RSimpleAttLayer(multi_embed, att_slayer,
                                      time_major=False,
                                      return_alphas=False)
    else:
        final_embed = tf.reduce_mean(multi_embed, axis=1, keep_dims=False)
    return final_embed


def forward_sage_fwt(inputs, ind, weights, reuse=False, dropout_rate=0.5, seed=1024, activation=tf.nn.relu, train_flag=None,
                 neighbor_num=[10, 10], agg='pooling', fwt=False, fwt_value=None):
    def hopagg(feat, weights, activation, fwt, fwt_value):
        node_feat, neigh_feat = feat
        node_feat_1 = Dropout(dropout_rate)(node_feat, training=train_flag)
        neigh_feat_1 = Dropout(dropout_rate)(neigh_feat, training=train_flag)
        concat_feat_1 = tf.concat([neigh_feat_1, node_feat_1], axis=1)
        concat_mean_1 = tf.reduce_mean(concat_feat_1, axis=1, keep_dims=False)
        output_1 = tf.matmul(concat_mean_1, weights[0])
        bais_output_1 = activation(output_1 + weights[1])

        if fwt:
            scale = fwt_value[0]
            shift = fwt_value[-1]
            output = tf.multiply(bais_output_1, scale) + shift
            return tf.identity(output)
        else:
            return bais_output_1

    def poolagg(feat, weights, activation, fwt, fwt_value):
        node_feat, neigh_feat = feat
        node_feat_1 = Dropout(dropout_rate)(node_feat, training=train_flag)
        neigh_feat_1 = Dropout(dropout_rate)(neigh_feat, training=train_flag)
        dims = tf.shape(neigh_feat_1)
        batch_size = dims[0]
        num_neighbors = dims[1]
        input_dim = dims[-1]
        h_reshaped = tf.reshape(
            neigh_feat, (batch_size * num_neighbors, input_dim))
        h_reshaped = tf.nn.relu(tf.matmul(h_reshaped, weights[2]) + weights[-1])
        neigh_feat = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, input_dim))
        neigh_feat_m = tf.reduce_mean(neigh_feat, axis=1, keep_dims=False)
        concat_feat = tf.concat([tf.squeeze(node_feat_1, axis=1), neigh_feat_m], axis=1)
        output_1 = tf.matmul(concat_feat, weights[0])
        bais_output_1 = activation(output_1 + weights[1])

        if fwt:
            scale = fwt_value[0]
            shift = fwt_value[-1]
            output = tf.multiply(bais_output_1, scale) + shift
            return output
        else:
            return bais_output_1


    def att_hopagg(feat, weights, activation, fwt, fwt_weight): 
        activation = tf.nn.elu
        node_feat, neigh_feat = feat
        node_feat_1 = Dropout(dropout_rate)(node_feat, training=train_flag)
        neigh_feat_1 = Dropout(dropout_rate)(neigh_feat, training=train_flag)
        node_feat_t = tf.tensordot(node_feat_1, weights[2], 1)
        neigh_feat_t = tf.tensordot(neigh_feat_1, weights[3], 1)

        node_feat_tile = tf.tile(node_feat_t, [1, neigh_feat_1.shape[1], 1])
        concat_feat = tf.concat(
            [neigh_feat_t, node_feat_tile], axis=2) 

        logits = tf.tensordot(concat_feat, weights[0], 1) 
        coefs = tf.nn.softmax(tf.nn.leaky_relu(
            tf.squeeze(logits, axis=-1)))  
        coefs = Dropout(dropout_rate)(coefs, training=train_flag)
        vals = tf.einsum('ij,ijk->ik', coefs, neigh_feat_t)
        ret = vals + weights[1]  
        bais_output_1 = ret + node_feat_t
        if activation:
            output = activation(ret)
        if fwt:
            gamma = fwt_weight[0]
            beta = fwt_weight[1]
            dim = gamma.get_shape().as_list()[-1]
            scale = 1 + tf.random_normal((1, dim), dtype=gamma.dtype) * tf.math.softplus(gamma)
            shift = tf.random_normal((1, dim), dtype=beta.dtype) * tf.math.softplus(beta)

            mean, variance = tf.nn.moments(bais_output_1, [-1], keep_dims=True)
            norm_bais_output_1 = (bais_output_1 - mean) / tf.sqrt(variance + 1e-5)
            output = tf.multiply(norm_bais_output_1, scale) + shift
            return output
        else:
            return bais_output_1


    if agg == 'mean':
        aggfun = hopagg
        single_sage_w_num = 2
    elif agg == 'pooling':
        aggfun = poolagg
        single_sage_w_num = 4
    elif agg == 'att':
        aggfun = att_hopagg
        single_sage_w_num = 4

    fwt_weights = []
    model_weights = []
    for i in weights:
        if 'fwt' in i.name:
            fwt_weights.append(i)
        else:
            model_weights.append(i)
    feature, node_input, hop1, hop2 = inputs

    node_input = tf.gather(node_input, ind)
    hop1 = tf.gather(hop1, ind)
    hop2 = tf.gather(hop2, ind)

    if fwt is True:
        scale = fwt_value[0]
        shift = fwt_value[-1]
        with tf.control_dependencies([]):
            ori_h = tf.multiply(feature, scale) + shift
    else:
        ori_h = feature
    hops = [node_input, hop1, hop2]
    input_feature_dim = 128
    n_hidden = 32
    for l in range(0, len(neighbor_num)): 
        after1agg = []
        if l == 0:
            feature_dim = input_feature_dim
        if l > 0:
            feature_dim = n_hidden
        for hop in range(len(neighbor_num) - l): 
            if l == 0:
                node_input_feat = tf.gather(ori_h, hops[hop])
                node_input_feat = tf.reshape(
                    node_input_feat, [-1, 1, feature_dim])
                hop1_feat = tf.gather(ori_h, hops[hop + 1])
                hop1_feat = tf.reshape(
                    hop1_feat, [-1, neighbor_num[hop], feature_dim])
            else:
                node_input_feat = mid_h[hop] 
                node_input_feat = tf.reshape(
                    node_input_feat, [-1, 1, feature_dim])
                hop1_feat = mid_h[hop + 1] 
                hop1_feat = tf.reshape(
                    hop1_feat, [-1, neighbor_num[hop], feature_dim]) 

            h = aggfun([node_input_feat, hop1_feat], model_weights[l *
                                                               single_sage_w_num:(l + 1) * single_sage_w_num],
                   activation, False, None)

            after1agg.append(h)
        mid_h = after1agg 
        after2hop = mid_h[0]

    return after2hop


from normalization import forward_modulate
def forward_han_sage_fwt(input_list, ind, weights, training_flag=None, reuse=False, mp_num=2, note='',
                     neighbor_num=[10, 25], sa='',ss=None, dropout=0.0, fwt=False, fwt_value=None, modulate=False):
    with tf.variable_scope('forward_fwt', reuse=reuse):
        hid_units = [32]
        activation = tf.nn.elu
   
        if ss is None:
            ss = []
        if 'ss' in ss:
            semantic = weights[4:]
        else:
            semantic = weights
        if sa == 'mlp':
            w_o = semantic[0]
            b_o = semantic[1]  
            num_sa_ = 2
        elif sa == 'sa':
            u_o = semantic[2]
            w_o = semantic[0]
            b_o = semantic[1] 
            num_sa_ = 3
        else:
            num_sa_ = 0
        if 'ss' in ss:
            bc_weights = weights[4+num_sa_:]
        else:
            bc_weights = weights[num_sa_:]
        modulate_weights = weights[-6:-2]

        dense_k = weights[-2]
        dense_b = weights[-1]

        ft = input_list[0]
        no = input_list[1]
        hop1s = input_list[2: 2 + mp_num]
        hop2s = input_list[2 + mp_num: 2 + mp_num * 2]

        embed_list = []
        for hop1, hop2 in zip(hop1s, hop2s):
            nb_nodes = tf.shape(ft)[0]
            mp_emb = forward_sage_fwt([ft, no, hop1, hop2], ind, bc_weights, dropout_rate=dropout, seed=1024,
                                  activation=activation,
                                  train_flag=training_flag, neighbor_num=neighbor_num, fwt=fwt, fwt_value=fwt_value)
            embed_list.append(tf.expand_dims(tf.reshape(
                mp_emb, (nb_nodes, hid_units[0])), axis=1))  

        multi_embed = tf.concat(embed_list, axis=1)
        if sa == 'mlp':
            final_embed = SimpleAttLayer(multi_embed, [w_o, b_o],
                                         time_major=False,
                                         return_alphas=False)
        elif sa == 'sa':
            final_embed = RSimpleAttLayer(multi_embed, [w_o, b_o, u_o],
                                          time_major=False,
                                          return_alphas=False)
        else:
            final_embed = tf.reduce_mean(multi_embed, axis=1)


        if modulate:
            modulate_mask = forward_modulate(final_embed, modulate_weights, activation, dropout, training_flag)
            final_embed = final_embed * modulate_mask
            logits = tf.tensordot(final_embed, dense_k, axes=1) + dense_b
        return logits, final_embed, []


def sample_neighs(G, nodes, sample_num=None, self_loop=False, shuffle=True): 
    _sample = np.random.choice
    neighs = [list(G[int(node)]) for node in nodes] 
    if sample_num:
        if self_loop:
            sample_num -= 1
        samp_neighs = [
            list(_sample(neigh, sample_num, replace=False)) if len(neigh) >= sample_num else list(
                _sample(neigh, sample_num, replace=True)) for neigh in neighs] 
        if self_loop:
            samp_neighs = [
                samp_neigh + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)] 

        if shuffle:
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
    else:
        samp_neighs = neighs
    return np.asarray(samp_neighs), np.asarray(list(map(len, samp_neighs)))


def sample_neighs_all_het(G, nodes, sample_num=None, self_loop=False, shuffle=True):  
    _sample = np.random.choice
    neighs = [list(G[int(node)]) for node in nodes]  

    if sample_num:
        if self_loop:
            sample_num -= 1
        samp_neighs = [
            (list(_sample(neigh, sample_num, replace=False)) if len(neigh) >= sample_num else list(
                _sample(neigh, sample_num, replace=True)))
            if len(neigh) > 0
            else list(nodes[i]) * sample_num
            for i, neigh in enumerate(neighs)] 

        if self_loop:
            samp_neighs = [
                samp_neigh + list([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)] 

        if shuffle:
            samp_neighs = [list(np.random.permutation(x)) for x in samp_neighs]
    else:
        samp_neighs = neighs
    return np.asarray(samp_neighs), np.asarray(list(map(len, samp_neighs)))
