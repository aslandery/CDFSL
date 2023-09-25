from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import networkx as nx
import tensorflow as tf
import numpy as np
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('/home/qnan/PyProjects/Multilabel data')
sys.path.append('/ibex/ai/home/zhanq0a/run00/mp')

sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
sys.path.append("....")
from maml_han_sage_fwt_batch_h2diver import GnnTrainModel
from han_sage_batch_fwt import sample_neighs_all_het

from domain_generators_het import source_domain_het, single_domain_het


def mp_node_3(iter, mp_adjs, N):
    mp = len(mp_adjs)
    assert mp > 1
    block_key = []
    for i in range(iter):
        node1 = []
        node2 = []
        label = []
        labels_ = {}
        for l, adj in enumerate(mp_adjs):
            rows, cols = adj.nonzero()
            num_non = len(rows)
            ind = np.random.choice(np.arange(num_non), N, replace=False)  
            rows = rows[ind]
            cols = cols[ind]

            for k1, k2 in zip(rows, cols):
                if (k1, k2) in labels_:
                    labels_[(k1, k2)][:,l] = 1
                    block_key.append((k1, k2))
                    del labels_[(k1, k2)]
                else:
                    if (k1, k2) in block_key:
                        continue
                    labels_[(k1, k2)] = np.zeros((1, mp))
                    labels_[(k1, k2)][:, l] = 1

        for key, value in labels_.iteritems():
            node1.append(key[0])
            node2.append(key[1])
            label.append(value)

        node1 = np.array(node1)
        node2 = np.array(node2)
        label = np.squeeze(np.stack(label), axis=1)
        output = np.stack((node1, node2), axis=-1)
        yield output, label


def mp_node_cd(iter, key,  N, p_list1, p_list2):
    for i in range(iter):
        p1 = np.random.choice(p_list1, N, replace=False)
        p2 = np.random.choice(p_list2, N, replace=False)
        pair = np.stack((p1, p2), axis=-1)
        yield pair


def mp_node_p(iter, mp_adjs, N, p_list, num):
    mp_adjs = mp_adjs[num:num + 1] 
    mp = len(mp_adjs)
    
    assert mp == 1
    block_key = []
    for i in range(iter):
        node1 = []
        labels_ = {}
        node2 = []
        label = []
        for l, adj in enumerate(mp_adjs):
            rows, cols = adj.nonzero()
            num_non = len(rows)
            ind = np.random.choice(np.arange(num_non), N, replace=False)  
            rows = rows[ind]
            cols = cols[ind]

            for k1, k2 in zip(rows, cols):
                labels_[(k1, k2)] = 1
        for key, value in labels_.iteritems():
            node1.append(key[0])
            node2.append(key[1])
            label.append(value)

        node1 = np.array(node1)
        node2 = np.array(node2)
        label = np.stack(label).reshape((-1, 1))
        output = np.stack((node1, node2), axis=-1)
        yield output, label


def mp_node(iter, mp_adjs, N, p_list, num):
    mp_adjs = mp_adjs[num:num + 1] 
    mp = len(mp_adjs)
    assert mp == 1
    block_key = []
    for i in range(iter):
        node1 = []
        labels_ = {}
        node2 = []
        label = []
        for l, adj in enumerate(mp_adjs):
            rows, cols = adj.nonzero()
            num_non = len(rows)
            ind = np.random.choice(np.arange(num_non), N, replace=False)  
            rows = rows[ind]
            cols = cols[ind]

            for k1, k2 in zip(rows, cols):
                labels_[(k1, k2)] = 1
	
        labelsr = {}
        for l, adj in enumerate(mp_adjs):
            num_c = 0
            rows, cols = adj.nonzero()
            while num_c < N:
                p1, p2 = np.random.choice(p_list, 2, replace=False)
		if ((p1, p2) not in labels_) and ((p2, p1) not in labels_):
		    labels_[(p1, p2)] = 0
                    num_c += 1
		
        labels_.update(labelsr)

        for key, value in labels_.iteritems():
            node1.append(key[0])
            node2.append(key[1])
            label.append(value)

        node1 = np.array(node1)
        node2 = np.array(node2)
        label = np.stack(label).reshape((-1, 1))
        output = np.stack((node1, node2), axis=-1)
        yield output, label


def early_stop_criteria(box,loss):
    if 'min' not in box:
        box['min'] = loss
        return False, box
    min_loss = box['min']
    patience = box['patience']
    if loss < min_loss:
        box['min'] = loss
        box['step_no'] = 0
    else:
        box['step_no'] += 1
        if box['step_no'] == patience:
            return True, box
    return False, box


def parse_args():
    parser = argparse.ArgumentParser(description="RUN ge parser.")
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--fix_test', help='tasks to test', action='store_true', default=False)
    parser.add_argument('--test_task', help='fix task to test', nargs='+', type=int, default=[4, 5])
    parser.add_argument('--meta_cv', help='fold for meta split', type=int, default=1)
    parser.add_argument('--hidden', help='hidden size for gnn', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--use_archive', action='store_true', default=False,
                        help='step of single task.')
    parser.add_argument('--use_archive_path', type=str, default='',
                        help='step of single task.')
    parser.add_argument('--sdataset', type=str, default='sys',
                        help='Dataset [delve, cora]')
    parser.add_argument('--edataset', type=str, default='sys',
                        help='Dataset [delve, cora]')
    parser.add_argument('--tdataset', type=str, default='ai',
                        help='Dataset [delve, cora]')
    parser.add_argument('--lr',  type=float, default=0.0005,
                        help='Learning rate [0.01, 0.001, 0.0001]')
    parser.add_argument('--inner_lr', type=float, default=0.05,
                        help='adapt for test class before evaluation')
    parser.add_argument('--update1', type=int, default=3,
                        help='adapt for test class before evaluation')
    parser.add_argument('--update2', type=int, default=5,
                        help='adapt for test class before evaluation')
    parser.add_argument('--nway', type=int, default=2,
                        help='N way . Default is 2.')
    parser.add_argument('--kshot', type=int, default=3,
                        help='k shot, Default is 5.')
    parser.add_argument('--kquery', type=int, default=15,
                        help='query points. Default is 10.')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='adapt for test class before evaluation')
    parser.add_argument('--nepoch', type=int, default=100,
                        help='different episodic task. Default is 128.')
    parser.add_argument('--iteration', type=int, default=10,
                        help='step of single task.')
    parser.add_argument('--pat', type=int, default=150, 
                        help='step of single task.')

    parser.add_argument('--pur', type=str, default='paper',
                        help='step of single task.')
    parser.add_argument('--mp', type=str, default='pap_pvp',
                        help='step of single task.')
    parser.add_argument('--module', type=str, default='ss_pap',
                        help='step of single task.')
    parser.add_argument('--bc', type=str, default='rgcn')
    parser.add_argument('--taskmode', type=str, default=None)

    parser.add_argument('--sa', default='mlp', type=str)
    parser.add_argument('--ss_alpha', type=float, default=0.3,
                        help='adapt for test class before evaluation')
    parser.add_argument('--randomf', help='tasks to test', action='store_true', default=False)
    parser.add_argument('--note_', type=str, default='',
                        help='step of single task.')
    parser.add_argument('--decay_steps', type=int, default=500,
                        help='step of single task.')
    parser.add_argument('--decay_rate', type=float, default=0.96,
                        help='step of single task.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='step of single task.')
    parser.add_argument('--sa_len', type=int, default=32,
                        help='step of single task.')
    parser.add_argument('--w_init', type=float, default=1,
                        help='adapt for test class before evaluation')
    parser.add_argument('--b_init', type=float, default=0,
                        help='adapt for test class before evaluation')
    parser.add_argument('--change', type=str, default='adapt',
                        help='adapt for test class before evaluation')
    parser.add_argument('--seed', type=int, default=2020,
                        help='adapt for test class before evaluation')
    return parser.parse_args()


def meta_train(args):

    nepochs = args.nepoch
    iteration = args.iteration
    nway = args.nway
    num_support_points = args.kshot
    num_query_points = args.kquery
    meta_batch_size = args.batch_size


    fast_updates1 = args.update1
    inner_lr = args.inner_lr
    meta_lr = args.lr
    fast_updates2 = args.update2

    meta_cv = args.meta_cv
    patience = args.pat

    module = args.module
    task_mode = args.taskmode

    cv = []
    neigh_number, feat_dim = [10, 25], 128

    for c in range(meta_cv):
        domain_labels, graph_datas, generators, slabel, elabel, adj_lists, node_lists = source_domain_het(args.pur, batchsize=meta_batch_size,
                                                                                   nway=nway,
                                                                                   num_support_points=num_support_points,
                                                                                   num_query_points=num_query_points,
                                                                                   edataset=args.edataset,
                                                                                   tdataset=args.tdataset,
                                                                                   neighbors=neigh_number,
                                                                                   mp=args.mp)
        _, eval_graph, eval_gen, _ = single_domain_het(args.pur, batchsize=1, nway=nway,
                                                    num_support_points=num_support_points,
                                                    num_query_points=num_query_points, elabel=elabel,
                                                    edataset=args.edataset, neighbors=neigh_number,
                                                    mp=args.mp)
        _, test_graph, test_gen,_ = single_domain_het(args.pur, batchsize=1, nway=nway,
                                                    num_support_points=num_support_points,
                                                    num_query_points=num_query_points, elabel=slabel,
                                                    edataset=args.tdataset, neighbors=neigh_number,
                                                    mp=args.mp)
        domain_ss = []
        cd_pair = {}
        for d in range(len(domain_labels)):
            adj_list = adj_lists[d]
            p_list, a_list, v_list = node_lists[d]
            if 'ss' in module:
                if 'pap' in module:
                    num = 0
                    ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, p_list, num)
                elif 'pvp' in module:
                   num = 1
                   ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, p_list, num)
                elif 'ppp' in module:
                    num = 2
                    ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, p_list, num)
                elif 'pp' in module:
                    num = 1
                    ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, p_list, num)
                elif 'pbp' in module:
                    num = 0
                    ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, p_list, num)
                else:
                    num = 0
                    ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, p_list, num)
                if 'apa' in module:
                    num = 0
                    ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, a_list, num)
                elif 'apvpa' in module:
                    num = 1
                    ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, a_list, num)
                elif 'aaa' in module:
                    num = 1
                    ss_batch_gen = mp_node_p(int(nepochs*iteration), adj_list, 32, a_list, num)
                domain_ss.append(ss_batch_gen)
                if args.pur == 'paper':
                    for d_y in range(len(domain_labels)):
                        keyp = 's' + str(d) + 't' + str(d_y)
                        p_list2, _, _ = node_lists[d_y]
                        domain_pair = mp_node_cd(int(nepochs*iteration),keyp,  32, p_list, p_list2)
                        cd_pair[keyp] = domain_pair
                elif args.pur == 'author':
                    for d_y in range(len(domain_labels)):
                        _, a_list2, _ = node_lists[d_y]
                        keyp = 's' + str(d) + 't' + str(d_y)
                        domain_pair = mp_node_cd(int(nepochs*iteration),keyp, 32, a_list, a_list2)
                        cd_pair.append(domain_pair)

        re_train = True         
        if nepochs == 0:
            re_train = False
        train_model = GnnTrainModel(neigh_number, nway=nway, kshot=num_support_points,hdims=[32],
                                    meta_batch_size=meta_batch_size,fast_updates=fast_updates1,
                                    inner_lr=inner_lr, meta_lr=meta_lr,
                                    feat_dim=feat_dim, mp_num=len(args.mp.split('_')), module=module,
                                    task_mode=task_mode,
                                    sa=args.sa, ss_alpha=args.ss_alpha, decay_steps=args.decay_steps,
                                    decay_rate=args.decay_rate, dropout=args.dropout, impo=None,
                                    sa_len=args.sa_len, w_init=args.w_init, b_init=args.b_init,
                                    random=args.randomf, bc=args.bc)
        train_model.build_opt('train', fast_updates2)

        dir = 'tmp/' + str(np.random.rand())[2:7] + '/'
        while os.path.exists(dir):
            dir = 'tmp/' + str(np.random.rand())[2:7] + '/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        best_saver = tf.train.Saver()
        best_dir = dir + 'best/'
        best_index = 0
        glob_init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        with tf.Session(config=config) as sess:
            sess.run(glob_init)
            if re_train:
                box = {'curr_step': 0, 'm1': 0.0, 'm2': np.inf, 'pat': patience}
              
                for e in range(nepochs):
                    for iter in range(iteration):
                        dom1, dom2 = np.random.choice(np.arange(3), 2, replace=False)
                        s_gen = generators[dom1]
                        t_gen = generators[dom2]
                        graph_s = graph_datas[dom1]
                        graph_t = graph_datas[dom2]
                        dic = dict()
                        train_tensor_dict = dict(zip(
                            train_model.sinput_tensor, graph_s))
                        train_tensor_dict.update(dict(zip(
                            train_model.tinput_tensor, graph_t)))
                        dic.update(train_tensor_dict)
                        s = s_gen.next()
                        t = t_gen.next()
                        sdic = {k: d for k, d in zip(train_model.s_input_list, s)}
                        tdic = {k: d for k, d in zip(train_model.t_input_list, t)}
                        dic.update(sdic)
                        dic.update(tdic)
                        ss_batch1, _ = next(domain_ss[dom1])
                        ss_batch2, _ = next(domain_ss[dom2])	
                        keyp = 's' + str(dom1) + 't' + str(dom2)
                        ss_batch3 = next(cd_pair[keyp])
                        inp = [ss_batch1, ss_batch2, ss_batch3]

                        ss_dic = {k: d for k, d in zip(train_model.ss_input, inp)}
                        dic.update(ss_dic)
                        dic[train_model.training_tensor] = 1
                        if args.change == 'mmd':
                            sess.run([train_model.mmd], feed_dict=dic)
                        elif args.change=='adapt':
                            sess.run([train_model.adaption], feed_dict=dic)
                    if e % 1 == 0:
                        mean_eval_loss = []
                        mean_eval_acc = []
                        for _ in range(50):
                            eval_d = eval_gen.next()
                            eval_tensor_dic = dict(zip(train_model.einput_tensor, eval_graph))
                            eval_dic = {k: d for k, d in zip(train_model.eval_input_list, eval_d)}
                            eval_dic.update(eval_tensor_dic)
                            eval_dic[train_model.training_tensor] = 0
                            eval_loss, eval_acc = sess.run([train_model.eval_total_losses2, train_model.eval_total_accuracies2], feed_dict=eval_dic)
                            mean_eval_acc.append(eval_acc)
                            mean_eval_loss.append(eval_loss)
                        mean_acc = np.mean(np.array(mean_eval_acc), axis=0)
                        mean_loss = np.mean(np.array(mean_eval_loss), axis=0)
                     
                        b1 = mean_acc[-1]
                        b2 = mean_loss[-1]
                        print('eval', mean_acc, mean_loss)
 			if b1 >= box['m1'] or b2 <= box['m2']:
                            if b1 >= box['m1'] and b2 <= box['m2']:
                                vacc_early_model = b1
                                vlss_early_model = b2
                                best_saver.save(sess, best_dir)
                            	box['m1'] = np.max((b1, box['m1'] ))
                            	box['m2'] = np.min((b2, box['m2']))
		            	best_index = e
                            box['curr_step'] = 0
			else:
                            box['curr_step'] += 1
                            if box['curr_step'] == box['pat'] :
                                
                                break


        if 1:
            imported_graph = tf.train.Saver()

            with tf.Session() as sess2:
                sess2.run(tf.global_variables_initializer())

                test_as = []
                test_ls = []
                if 1:
                    if args.use_archive is False:
                        imported_graph.restore(sess2, best_dir)
                    else:
                        archive_dir = args.use_archive_path
                        imported_graph.restore(sess2, archive_dir)

                for _ in range(50):
                    t_task = test_gen.next()
                    test_tensor_dic = dict(zip(train_model.einput_tensor, test_graph))
                    test_dic = {k: d for k, d in zip(train_model.eval_input_list, t_task)}
                    test_dic.update(test_tensor_dic)
                    test_dic[train_model.training_tensor] = 0
                    test_loss, test_acc = sess2.run([train_model.eval_total_losses2, train_model.eval_total_accuracies2], feed_dict=test_dic)
                    test_as.append(test_acc)
                    test_ls.append(test_loss)
                mean_test_acc = np.mean(np.array(test_as), axis=0)
              
        cv.append(mean_test_acc)
        tf.reset_default_graph()

    output_args = [nepochs, iteration, meta_lr, inner_lr, fast_updates2, nway, num_support_points, args.pur, args.mp]
    filename = ('_').join([str(i) for i in output_args]) + '_r' + str(np.random.rand())[2:7]
    with open('output/' + filename + '.txt', 'w') as f:
        
        res = 'Total Meta-Test Accuracy across cv{0:5d}'.format(meta_cv)
        f.write(res)
        f.write('\n')
        f.write(str(np.mean(cv, axis=0)))
        f.write('\n')




def main():
    args = parse_args()
    glo_seed = args.seed

    os.environ['PYTHONHASHSEED'] = str(glo_seed)
    np.random.seed(int(glo_seed))
    tf.set_random_seed(int(glo_seed))
    meta_train(args)

if __name__ == '__main__':

    main()

