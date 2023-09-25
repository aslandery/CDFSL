import sys
sys.path.append('.')
sys.path.append('..')
from datahelper_het import *


def single_domain_het(pur, batchsize, nway, num_support_points, num_query_points, elabel, edataset, neighbors, mp):
    def load_dataloader(dataset, label, pur, batchsize):
        dataloader = SDomainNeighborLoader(nway, num_support_points, num_query_points, neighbors)
        wholelabels, input_tensor, adj, nodelist  = dataloader.loader(dataset, label, mp, pur)
        s_gen = dataloader.batch(batchsize)
        return wholelabels, input_tensor, s_gen, adj, nodelist

    wholelabels, input_, generator_, adj, nodelist = load_dataloader(edataset, elabel[edataset], pur, batchsize)

    return elabel[edataset], input_, generator_, wholelabels


def source_domain_het(pur, batchsize, nway, num_support_points, num_query_points, edataset, tdataset, neighbors, mp):
    def load_dataloader(dataset, label, pur, batchsize):
        dataloader = SDomainNeighborLoader(nway, num_support_points, num_query_points, neighbors)
        _, input_tensor, adj, nodelist = dataloader.loader(dataset, label, mp, pur)

        s_gen = dataloader.batch(batchsize)
        return input_tensor, s_gen, adj, nodelist

    def load_label(edataset):
        if edataset == 'sys':
            cats = np.arange(11)
            np.random.shuffle(cats)
            sys_label = cats[:7]
            sdataset_label = {'sys': sys_label, 'ai': np.arange(8), 'math': np.arange(8), 'inter': np.arange(7)}
            edataset_label = {'sys': list(set(np.arange(11)) - set(sys_label))}
        if edataset == 'ele':
            cats = np.arange(14)
            np.random.shuffle(cats)
            sys_label = cats[:9]
            sdataset_label = {'ele': sys_label, 'che': np.arange(7), 'mat': np.arange(7), 'com': np.arange(10)}
            edataset_label = {'ele': list(set(np.arange(14)) - set(sys_label))}
        if edataset == 'office':
            cats = np.arange(116)
            np.random.shuffle(cats)
            sys_label = cats[:70]
            sdataset_label = {'office': sys_label, 'arts': np.arange(93), 'music': np.arange(94), 'patio': np.arange(106), 'toy':np.arange(89)} # 'pet':np.arange(101)}
            edataset_label = {'office': list(set(np.arange(116)) - set(sys_label))}
  

        return sdataset_label, edataset_label
    if edataset == 'office':
	domains = ['office', 'arts', 'music', 'patio', 'toy']
    if edataset == 'ele':
        domains = ['ele', 'che', 'mat', 'com']    	
    if edataset == 'sys':
	domains = ['sys', 'ai', 'math', 'inter']
    slabel, elabel = load_label(edataset)
    source_domains = [d for d in domains if d != tdataset]

    generators = []
    graph_datas = []
    domain_labels = []
    adj_lists, node_lists= [],[]

    for s in source_domains:
        domain_labels.append(slabel[s])
        input_, generator_, adj, nodelist = load_dataloader(s, slabel[s], pur, batchsize)
        graph_datas.append(input_)
        generators.append(generator_)
        adj_lists.append(adj)
        node_lists.append(nodelist)
    return domain_labels, graph_datas, generators, slabel, elabel, adj_lists, node_lists

