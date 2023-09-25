import numpy as np
import networkx as nx


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


def sample_mp_neighbors(adj_list, neigh_number):
    mp_neighbor = []
    for adj in adj_list:   
        G = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
        indexs = np.arange(adj.shape[0]).reshape((-1, 1))   
        sample_neigh_2 = []
        neigh_maxlen = []
        for num in neigh_number:
            sample_neigh, sample_neigh_len = sample_neighs_all_het(
                G, indexs, num, self_loop=False)
            sample_neigh_2.append(sample_neigh)
            neigh_maxlen.append(max(sample_neigh_len))
        mp_neighbor.append(sample_neigh_2)
    flatten_list = []
    for i in range(len(adj_list)):  
        for j in range(len(neigh_number)): 
            flatten_list.append(mp_neighbor[i][j])
    return flatten_list  


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


def sampling(src_nodes, sample_num, neighbor_table):   
    results = []
    for sid in src_nodes:
        if len(neighbor_table[sid]) == 0:
            neighbor_table[sid].append(sid)
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))  
        results.append(res)  
    return np.asarray(results).flatten()  


def adj2table(G, nodes):
    _sample = np.random.choice
    neighs = [list(G[int(node)]) for node in nodes]   
    return neighs


def mix_rel_sampling(G, allG, src_nodes, sample_nums):
    neighlist1 = adj2table(G, src_nodes)
    neighlist2 = [adj2table(g, src_nodes) for g in allG]
    sampling_result = [src_nodes]  
    layer1G = sampling(sampling_result[0], sample_nums[0], neighlist1)
    sampling_result.append(layer1G)  

    layer2allG = []
    for n in neighlist2:
        layer2rel = sampling(sampling_result[1], sample_nums[1], n)

        layer2allG.append(layer2rel)
    return layer1G, layer2allG


from setuptools.namespaces import  flatten
def sample_1rgcn_hops(adj_list, neigh_number):
    assert len(adj_list)==1
    mp = len(adj_list)
    Gs = []
    adj_i = adj_list[0]
	
    G = nx.from_scipy_sparse_matrix(adj_i)
    Gs.append(G)
    source1 = np.arange(adj_i.shape[0])
    source1length = len(source1)
    
    layer1rels = []
    layer2rels = []
    for i in range(mp):
    	i_rel_l1, i_rels_l2 = mix_rel_sampling(Gs[i], Gs, source1, neigh_number)
	print(i_rel_l1[0], i_rels_l2[0])
        layer2rels.append(i_rels_l2)
        layer1rels.append(i_rel_l1)
    mix1 = np.reshape(i_rel_l1, [-1, neigh_number[0]])
    mix2 = np.reshape(i_rels_l2, [-1, neigh_number[0]* neigh_number[1]])
    return mix1, mix2

	
def sample_2rgcn_hops(adj_list, neigh_number):
    assert len(adj_list) == 2
    mp = len(adj_list)
    Gs = []
    for adj_i in adj_list:
        G = nx.from_scipy_sparse_matrix(adj_i)
        Gs.append(G)
        source1 = np.arange(adj_i.shape[0])
        source1length = len(source1)
    layer1rels = []
    layer2rels = []
    for i in range(mp):
        i_rel_l1, i_rels_l2 = mix_rel_sampling(Gs[i], Gs, source1, neigh_number)
        layer2rels.append(i_rels_l2)
        layer1rels.append(i_rel_l1)
 
    layer1rel1 = np.reshape(np.array(layer1rels[0]), [-1, neigh_number[0]]) 
    layer1rel2 = np.reshape(np.array(layer1rels[1]), [-1, neigh_number[0]])
    mix1 = np.concatenate((layer1rel1, layer1rel2), axis=-1)
    rel1rel1, rel1rel2, rel2rel1, rel2rel2 = list(flatten(layer2rels))
    rel1rel1 = np.reshape(np.array(rel1rel1), [-1, neigh_number[0], neigh_number[1]])
    rel1rel2 = np.reshape(np.array(rel1rel2), [-1, neigh_number[0], neigh_number[1]])
    rel1mix = np.concatenate((rel1rel1, rel1rel2), axis=-1)
    rel2rel1 = np.reshape(np.array(rel2rel1), [-1, neigh_number[0], neigh_number[1]])
    rel2rel2 = np.reshape(np.array(rel2rel2), [-1, neigh_number[0], neigh_number[1]])
    rel2mix = np.concatenate((rel2rel1, rel2rel2), axis=-1)
    mix2 = np.concatenate((rel1mix, rel2mix), axis=1)
    mix2 = np.reshape(mix2, [-1, neigh_number[0]*2*neigh_number[1]*2])

    return mix1, mix2


def sample_rgcn_hops(adj_list, neigh_number):
    if len(adj_list) == 1:
	mix1, mix2 = sample_1rgcn_hops(adj_list, neigh_number)
    elif len(adj_list) ==2:
        mix1, mix2 = sample_2rgcn_hops(adj_list, neigh_number)
    return mix1, mix2 



def sample_mp_neighbors_hops(adj_list, neigh_number):
    mp_sample_results = []  
    for adj in adj_list:  
        if isinstance(adj, list):
            G = []
            for adj_i in adj:
                G.append(nx.from_scipy_sparse_matrix(adj_i, create_using=nx.DiGraph()))
                source1 = np.arange(adj_i.shape[0])   
        else:
            G = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
            source1 = np.arange(adj.shape[0])   
        source_len = len(source1)
        sample_results = multihop_sampling(G, source1, neigh_number)
        mp_sample_results.append(sample_results)
    flatten_list = []
    for i in range(len(neigh_number)):  
        for j in range(len(adj_list)):   
            flatten_list.append(mp_sample_results[j][i+1].reshape([source_len,-1]))
    return flatten_list  


def multihop_sampling(Gs, src_nodes, sample_nums):
    neighbor_list = []
    if isinstance(Gs, list):
        for G_i in Gs:
            neighbor_list.append(adj2table(G_i, src_nodes))
    else:
        for _ in sample_nums:
            neighbor_list.append(adj2table(Gs, src_nodes))

    sampling_result = [src_nodes]   

    for k, hopk_num in enumerate(sample_nums):  
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_list[k])
        sampling_result.append(hopk_result)  

    return sampling_result
