import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.manifold import TSNE
import os
import pickle as pkl
import sys
from scipy.sparse import *
from sklearn.preprocessing import Normalizer
import scipy.io as sio
import time
import tensorflow as tf


def get_type_set(type, dataset, label_set):
    ss_p, ss_a, ss_v = get_type_index(dataset)
    type_label_set = {}
    if type == 'paper':
        for key in label_set:
            intersect = list(set(label_set[key]).intersection(set(ss_p)))
            type_label_set[key] = intersect
    elif type == 'author':
        for key in label_set:
            intersect = list(set(label_set[key]).intersection(set(ss_a)))
            type_label_set[key] = intersect
    else:
        type_label_set = label_set
    label_set = type_label_set
    return label_set


def get_splits(y,):
    idx_list = np.arange(len(y))
    idx_train = []
    label_count = {}
    for i, label in enumerate(y):
        label = np.argmax(label)
        if label_count.get(label, 0) < 20:
            idx_train.append(i)
            label_count[label] = label_count.get(label, 0) + 1
    idx_val_test = list(set(idx_list) - set(idx_train))
    idx_val = idx_val_test[0:500]
    idx_test = idx_val_test[500:1500]

    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    val_mask = sample_mask(idx_val, y.shape[0])
    test_mask = sample_mask(idx_test, y.shape[0])

    return y_train, y_val, y_test, train_mask, val_mask, test_mask  

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset_str):
    FILE_PATH = os.path.abspath(__file__)
    DIR_PATH = os.path.dirname(FILE_PATH)
    DATA_PATH = os.path.join(DIR_PATH, 'data/')
    DATA_PATH = "../data/cora/"
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATA_PATH, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATA_PATH, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return sp.csr_matrix(adj), features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def reverse_sample_mask(idx, l):
    mask = np.ones(l)
    mask[idx] = 0
    return np.array(mask, dtype=np.bool)


def convert_symmetric(X, sparse=True):
    if sparse:
        X += X.T - sp.diags(X.diagonal())
    else:
        X += X.T - np.diag(X.diagonal())
    return X


def encode_onehot(labels): 
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)  
    return labels_onehot


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def get_label_set_v1(dataset="cora", path="../data/cora/",):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    onehot_labels = encode_onehot(idx_features_labels[:, -1])  
    label_set = {}
    for i in range(len(onehot_labels)):
        if np.sum(onehot_labels[i]) > 0:
            label = np.argmax(onehot_labels[i])
            if label_set.has_key(label):
                label_set[label].append(i)
            else:
                label_set[label] = []
                label_set[label].append(i)

    return label_set


def load_data_v1(dataset="cora", path="../data/cora/",):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    onehot_labels = encode_onehot(idx_features_labels[:, -1]) 
  
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                        shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
    adj = convert_symmetric(adj, )

    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(onehot_labels)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, onehot_labels


def get_masked_label_set_v2(onehot_labels, mask):
    label_set = {}
    for i in range(len(onehot_labels)):
        if mask[i] is True:
            continue
        if np.sum(onehot_labels[i]) > 0:
            label = np.argmax(onehot_labels[i])
            if label_set.has_key(label):
                label_set[label].append(i)
            else:
                label_set[label] = []
                label_set[label].append(i)
    return label_set

def get_label_set_v2(onehot_labels):
    label_set = {}
    for i in range(len(onehot_labels)):
        if np.sum(onehot_labels[i]) > 0:
            label = np.argmax(onehot_labels[i])
            if label_set.has_key(label):
                label_set[label].append(i)
            else:
                label_set[label] = []
                label_set[label].append(i)
    return label_set


def load_data_v2(dataset, emb_type='lsi'):
    node_features = load_npz(
        "../{}/{}_matrices/{}.npz".format(dataset, dataset, emb_type)).toarray()
    node_features = Normalizer().fit_transform(node_features)  
    labels = sp.load_npz("../{}/{}_matrices/label.npz".format(dataset, dataset)).astype(np.int32)  
    edges = np.genfromtxt("../{}/{}.edgelist".format(dataset, dataset), dtype=np.int32)   
    node_num = node_features.shape[0]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  
                        shape=(node_num, node_num), dtype=np.float32)  
    adj = convert_symmetric(adj, )
    return adj, node_features, labels


def load_data_v3(dataset, emb_type='lsi'):

    node_features = np.load(
        "../{}/{}_embeddings.npz".format(dataset, dataset))
    node_features = Normalizer().fit_transform(node_features['arr_0'])   
    labels = sp.load_npz("../{}/{}_labels.npz".format(dataset, dataset)).astype(np.int32)   
    f = open("../{}/int_{}.edgelist".format(dataset, dataset),'r')
    edges = [[l.split(',')[0], l.split(',')[1]] for l in f]
    edges = np.array(edges)
    edge_num = len(edges)
    node_num = node_features.shape[0]
    adj = sp.coo_matrix((np.ones(edge_num), (edges[:, 0], edges[:, 1])),  
                        shape=(node_num, node_num), dtype=np.float32)  
    adj = convert_symmetric(adj, )
    return adj, node_features, labels


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])   
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1]))) 
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt) 


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes) 
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack(
        (adj.col, adj.row)).transpose()
  
    return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape) ##


def load_data_ss(dataset):
    ss_labels = sp.load_npz("../{}/{}_ss_labels.npz".format(dataset, dataset)).astype(np.int32)  
    return ss_labels


def get_type_index(dataset):
    ss_labels = sp.load_npz("../{}/{}_ss_labels.npz".format(dataset, dataset)).astype(np.int32) 
    ssl = ss_labels.todense()
    p_type = []
    a_type = []
    v_type = []

    for i in range(ssl.shape[0]):
        if ssl[i, 0] == 1:
            p_type.append(i)
        elif ssl[i, 1] == 1:
            a_type.append(i)
        else:
            assert ssl[i, 2] ==1
            v_type.append(i)
    return p_type, a_type, v_type


def load_data_het(dataset, mp, emb_type='lsi'):
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
                  t_task=list(), proto='proto'):
   

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


def main():
    load_data_v2('cora')

if __name__ == '__main__':
    main()
