import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import networkx as nx
import scipy.sparse as sp

from sklearn.preprocessing import Normalizer
import scipy.io as sio


from utils import get_label_set_v2, load_data_ss, get_type_set, get_type_index, preprocess_adj
from hops import sample_mp_neighbors_hops
from sage import sample_neighs
from hops import sample_rgcn_hops


def process_data_rgcn(dataset, mp, pur, neighbor_num):
    adj_list, fea_list, total_labels, mp_num = load_data_het(dataset, mp)
    nb_nodes = fea_list[0].shape[0]
    feat_dim = fea_list[0].shape[1]
    total_labels = total_labels.toarray()
    label_set = get_label_set_v2(total_labels)

    if pur is not '':
        ss_labels = load_data_ss(dataset)
        ss_labels = ss_labels.toarray()
        ss_label_set = get_label_set_v2(ss_labels)
        label_set = get_type_set(pur, dataset, label_set)
    p_list, a_list, v_list = get_type_index(dataset)
  
    neigh_list = sample_rgcn_hops(adj_list, neighbor_num)
    indexs = np.arange(nb_nodes).reshape((-1, 1)) 
    input_v = []
    input_v.append(fea_list[0])
    input_v.append(indexs)
    input_v.extend(neigh_list)
    nodelist = [p_list, a_list, v_list]
    return label_set, input_v, mp_num, adj_list, nodelist


def load_data_het(dataset, mp):
    node_features = np.load(
        "../{}/{}_embeddings.npz".format(dataset, dataset))
    node_features = Normalizer().fit_transform(node_features['arr_0'])  

    labels = sp.load_npz("../{}/{}_labels.npz".format(dataset, dataset)).astype(np.int32)  

    data = sio.loadmat('../{}/mp_adjs.mat'.format(dataset))

    if 'adj' not in mp:
        mps = mp.split('_')
        rownetworks = []
        truefeatures_list = []
        for mp in mps:
            if '+' in mp:
                semi_mps = mp.split('+')
                semi_adjs = []
                for semi_mp in semi_mps:
                    semi_adjs.append(data[semi_mp])
                m_adj = semi_adjs
            else:
                m_adj = data[mp]

            rownetworks.append(m_adj)
            truefeatures_list.append(node_features)

    f = open("../{}/int_{}.edgelist".format(dataset, dataset),'r')
    edges = [[l.split(',')[0], l.split(',')[1]] for l in f]

    edges = np.array(edges)
    edge_num = len(edges)
    node_num = node_features.shape[0]
    adj = sp.coo_matrix((np.ones(edge_num), (edges[:, 0], edges[:, 1])),  
                        shape=(node_num, node_num), dtype=np.float32)   
    if 'adj' in mp:
        rownetworks = [adj]
        truefeatures_list = [node_features]

    return rownetworks, truefeatures_list, labels, len(rownetworks)


def prepare_nk_q(label_set, n, k, q,
                  t_task=list(), proto='mlp'):
    ndim = k + q
    if len(t_task) > n:
        task = np.random.choice(t_task, n, False)
    else:
        task = t_task

    supsdata = []
    quesdata = []
    for i in range(n):
        data = np.random.choice(label_set[task[i]], ndim, False)
        label = np.full((1, ndim), i)
        waydata = np.hstack((data.reshape(ndim, -1), label.reshape(ndim, -1)))
        supdata = waydata[:k, ]  
        quedata = waydata[k:, ] 
        supsdata.extend(supdata) 
        quesdata.extend(quedata)  

    supsdata = np.array(supsdata)  
    quesdata = np.array(quesdata)  

    if proto is 'mlp':
        np.random.shuffle(supsdata) 
    np.random.shuffle(quesdata)
    support = supsdata[:, 0]
    support_l = np.eye(n)[supsdata[:, -1]]
    query = quesdata[:, 0]
    query_l = np.eye(n)[quesdata[:, -1]]

    return support, support_l, query, query_l


def process_data(dataset, mp, pur, neighbor_num):
    adj_list, fea_list, total_labels, mp_num = load_data_het(dataset, mp)
    nb_nodes = fea_list[0].shape[0]
    feat_dim = fea_list[0].shape[1]
    total_labels = total_labels.toarray()
    label_set = get_label_set_v2(total_labels)

    if pur is not '':
        ss_labels = load_data_ss(dataset)
        ss_labels = ss_labels.toarray()
        ss_label_set = get_label_set_v2(ss_labels)
        label_set = get_type_set(pur, dataset, label_set)

    p_list, a_list, v_list = get_type_index(dataset)
    neigh_list = sample_mp_neighbors_hops(adj_list, neighbor_num)

    indexs = np.arange(nb_nodes).reshape((-1, 1)) 
    indexes_list = [indexs for _ in range(mp_num)]
    input_v = []
    input_v.append(fea_list[0])
    input_v.append(indexs)
    input_v.extend(neigh_list)
    nodelist = [p_list, a_list, v_list]
    return label_set, input_v, mp_num, adj_list, nodelist


class SDomainNeighborLoader(object):
    def __init__(self, nway, support, query, neighbor):
        self.nway = nway
        self.support = support
        self.query = query
        self.neighbor = neighbor

    def loader(self, dataset, label, mp, pur):
        self.label = label
        self.label_set, self.input_tensor, _, adj_list, nodelist = process_data_rgcn(dataset, mp,  pur, self.neighbor)
        return self.label_set, self.input_tensor, adj_list, nodelist

    def batch(self, meta_batch_size, proto='mlp'):
        while True:
            sup_batch = []
            sup_l_batch = []
            query_batch = []
            query_l_batch = []
            kshots = []
            for i in range(meta_batch_size):
                task = np.random.choice(self.label, self.nway, replace=False)
                sup, sup_l, query, query_l = prepare_nk_q(self.label_set, self.nway, self.support, self.query,
                                                          t_task=task, proto=proto)
                sup_batch.append(sup)
                sup_l_batch.append(sup_l)
                query_batch.append(query)
                query_l_batch.append(query_l)
            kshots += sup_batch
            kshots += sup_l_batch
            kshots += query_batch
            kshots += query_l_batch
            yield kshots

