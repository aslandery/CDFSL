from utils import *
from tensorflow.python.keras import backend as K
from han_sage_batch_fwt import GraphSAGE_fwt, forward_han_sage_fwt
from utils_mlp import OneLayerClassifier
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from maml_task import BaseTask
from normalization import  Modulate
from rgcn_sage_concat import GraphSAGE_rgcn, forward_rgcn
from normalization import  Reg, forward_reg
class GnnTrainModel( BaseTask):
    def __init__(self, neighbor_num, nway,kshot, hdims, meta_batch_size,fast_updates, inner_lr, meta_lr,
                 feat_dim, mp_num, module=None, task_mode=None, sa=False, ss_alpha=0.3, kalpha = 1,  decay_steps=500,decay_rate=0.96,
                 dropout=0.0, impo='degree', sa_len=64, w_init=1, b_init=0, random=False, bc=None):

        self.nway = nway
        self.meta_batch_size=meta_batch_size
        self.test_batch_size = 1
        self.mapped_dim_y = nway
        self.kshot = kshot
        self.num_grad_updates = fast_updates
        self.inner_lr = inner_lr
        self.lr = meta_lr

        self.dropout = dropout
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.hdims = hdims
        self.neighbor_num = neighbor_num
        self.mp_num = mp_num
        self.feature_dim = feat_dim

        self.model = bc
        self.impo = impo
        self.ss_alpha = ss_alpha
        self.ss_way = 3
        self.module = module  
        self.task_mode = task_mode  
        self.sa = sa
        self.sa_size = sa_len

        self.stype = tf.float32
        self.inttype = tf.int32
        self.training_tensor = tf.placeholder(tf.bool)
        self.w_init = w_init
        self.b_init = b_init
        self.feature_random = False
        self.kalpha = kalpha
        self.get_ss_model()

    def get_mlloss(self, logits, train_label):
        loss_ = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(train_label, self.stype), logits=logits)
        loss_ = tf.reduce_mean(loss_)
        return loss_

    def get_loss(self, logits, train_label):
        loss_ = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(train_label, self.inttype), logits=logits, dim=-1)
        loss_ = tf.reduce_mean(loss_)
        return loss_

    def get_acc(self, logits, train_label):
        arg_y = tf.cast(tf.argmax(logits, -1), self.stype)   
        actual = tf.cast(tf.argmax(train_label, axis=-1), self.stype)   
        with tf.control_dependencies([]):
            corr = tf.equal(actual, arg_y)
            acc = tf.reduce_mean(tf.cast(corr, self.stype))
        return acc

    def build_opt(self, mode, update_num):
        naive_, couple_, super_ = False, True, False

        if mode == 'test':
            self.build_target_input(1, mode)
            self.get_model()
            self.opt_eval(self.eval_input_list, 1, update_num)
        else:
            self.build_input(self.meta_batch_size, mode)
            self.build_target_input(batch=self.meta_batch_size, mode=mode)  
            self.get_model()

            if 'ss' in self.module:
                self.build_ss_input(mode)
                self.opt(self.s_input_list, self.t_input_list, self.meta_batch_size, mode, self.num_grad_updates, ss_inp=self.ss_input,
                         tar_inp=None, naive_setting=naive_, couple_setting=couple_, super_setting=super_)
            else:
                self.opt(self.s_input_list, self.t_input_list, self.meta_batch_size, mode, self.num_grad_updates,
                         tar_inp=None, naive_setting=naive_, couple_setting=couple_, super_setting=super_)

            self.opt_eval(self.eval_input_list, 1, update_num)

    def build_tensor(self, name):
        input_tensor = []
        with tf.name_scope('input'):
            with tf.device('/cpu:0'):
                features_tensor = tf.placeholder(shape=(None, self.feature_dim), dtype=self.stype, name=name)
                node_tensor = tf.placeholder(shape=(None, 1), dtype=self.inttype, name=name)
                hop1_tensor = tf.placeholder(shape=(None, self.mp_num * self.neighbor_num[0]), dtype=self.inttype, name=name)

                hop2_tensor = tf.placeholder(shape=(None, self.mp_num * self.neighbor_num[0] * self.mp_num * self.neighbor_num[1]), dtype=self.inttype, name=name)
                input_tensor.append(features_tensor)
                input_tensor.append(node_tensor)
                input_tensor.append(hop1_tensor)
                input_tensor.append(hop2_tensor)

            return input_tensor


    def build_tensor_(self, name):
        input_tensor = []
        with tf.name_scope('input'):
            with tf.device('/cpu:0'):
                features_tensor = tf.placeholder(shape=(None, self.feature_dim), dtype=self.stype, name=name)
                node_tensor = tf.placeholder(shape=(None, 1), dtype=self.inttype, name=name)
                hop1_tensor = [tf.placeholder(shape=(None, 1 * self.neighbor_num[0]), dtype=self.inttype, name=name) for i in range(self.mp_num)]
                hop2_tensor = [tf.placeholder(shape=(None, 1 * self.neighbor_num[1]), 
                                              dtype=self.inttype, name=name) for i in range(self.mp_num)]
                input_tensor.append(features_tensor)
                input_tensor.append(node_tensor)
                input_tensor.extend(hop1_tensor)
                input_tensor.extend(hop2_tensor)
            return input_tensor

    def build_input(self, batch, mode, eval_batch=1):
        if mode == 'train':
            obs_vars, lab_vars = self.make_vars(batch, 'train_s') 
            obs_q_vars, lab_q_vars = self.make_vars(batch, 'train_q')
            self.s_input_list = []
            self.s_input_list += obs_vars + lab_vars + obs_q_vars + lab_q_vars
            self.sinput_tensor = self.build_tensor('sinput')

    def build_target_input(self, batch, mode, eval_batch=1):
        if mode == 'train':
            obs_vars, lab_vars = self.make_vars(batch, 'tar_s')  
            obs_q_vars, lab_q_vars = self.make_vars(batch, 'tar_q')
            self.t_input_list = []
            self.t_input_list += obs_vars + lab_vars + obs_q_vars + lab_q_vars
            self.tinput_tensor = self.build_tensor('tinput')

            eval_obs_vars, eval_lab_vars = self.make_vars(eval_batch, 'eval_s')  
            eval_q_vars, eval_q_lab = self.make_vars(eval_batch, 'eval_q') 
            self.eval_input_list = []
            self.eval_input_list += eval_obs_vars + eval_lab_vars + eval_q_vars + eval_q_lab
            self.einput_tensor = self.build_tensor('eval')

    def build_ss_input(self, mode):
        if mode == 'train':
            self.ss_input = []
            obs1 = tf.placeholder(dtype=self.inttype, shape=(None,2), name='ss_obs1')
            lab1 = tf.placeholder(dtype=self.inttype, shape=(None, 1), name='ss_lab1')
            self.ss_input.append(obs1)

            obs2 = tf.placeholder(dtype=self.inttype, shape=(None,2), name='ss_obs2')
            lab2 = tf.placeholder(dtype=self.inttype, shape=(None, 1), name='ss_lab2')
            self.ss_input.append(obs2)

            obs3 = tf.placeholder(dtype=self.inttype, shape=(None,2), name='ss_obs3')
            lab3 = tf.placeholder(dtype=self.inttype, shape=(None, 1), name='ss_lab3')
            self.ss_input.append(obs3)
	
    def get_model(self):
        with tf.variable_scope('backbone', reuse=tf.AUTO_REUSE):
            if self.model == 'rgcn':
                emb_model = GraphSAGE_rgcn(input_list=self.sinput_tensor,
                                              feature_dim=self.feature_dim,
                                              neighbor_num=self.neighbor_num,
                                              n_hidden=self.hdims[0],
                                              n_classes=self.hdims[-1],
                                              use_bias=True,
                                              activation=tf.nn.relu,
                                              dropout_rate=self.dropout,
                                              l2_reg=2.5e-4,
                                              training=self.training_tensor,
                                              mp_num=self.mp_num)

            else:
                emb_model = GraphSAGE_fwt(input_list=self.sinput_tensor,
                              input_feature_dim=self.feature_dim,
                              neighbor_num=self.neighbor_num,
                              n_hidden=self.hdims[0],
                              n_classes=self.hdims[-1], 
                              use_bias=True,
                              activation=tf.nn.relu,
                              aggregator_type='pooling',
                              dropout_rate=self.dropout,
                              l2_reg=2.5e-4,
                              training=self.training_tensor,
                              mp_num=self.mp_num,
                              stype=self.stype,
                              sa=self.sa,
                              sa_size=self.sa_size,
                              fwt=True,
                              w_init=self.w_init,
                              b_init=self.b_init)

            modulate_model = Reg(32,
                 activation=tf.nn.relu, dropout_rate=self.dropout,
                 use_bias=True, l2_reg=2.5e-4,
                 seed=1024)(emb_model)
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            classifier_model = OneLayerClassifier(units=self.nway)(emb_model)

        self.weights = tf.trainable_variables()
        self.model_params, self.reg_params = [], []
        self.cls_params = []
        self.fea_params = []
        for var in self.weights:
            if 'regloss' not in var.name:
                self.model_params.append(var)
            else:
                self.reg_params.append(var)
        for var in self.model_params:
            if 'class' in var.name:
                self.cls_params.append(var)
            else:
                self.fea_params.append(var)
        return classifier_model

    def get_ss_model(self):  
        with tf.variable_scope('on-aux', reuse=tf.AUTO_REUSE):
            mp_attr = self.hdims[-1]
            mid_hid = 32
            out_dim = 16
            mp_weight = tf.get_variable('mp_kernel', (mp_attr, mid_hid), initializer=glorot_uniform(), dtype=self.stype)
            mp_bias = tf.get_variable('mp_bias', (mid_hid), initializer=Zeros(), dtype=self.stype)
            mp_weight2 = tf.get_variable('mp_kernel2', (mid_hid, out_dim ), initializer=glorot_uniform(), dtype=self.stype)
            mp_bias2 = tf.get_variable('mp_bias2', (out_dim), initializer=Zeros(), dtype=self.stype)

    def opt_eval(self, inp,  meta_size, update_num):
        def batch_input(inp, meta_size):
            elems = []
            for i in range(meta_size):  
                elem = []
                elem.append(inp[i]) 
                elem.append(inp[1 * meta_size + i])
                elem.append(inp[2 * meta_size + i])
                elem.append(inp[3 * meta_size + i]) 
                elems.append(elem)
            return elems

        def fast_params(symbol_loss, symbol_params, lr=None):
            gvs = tf.gradients(symbol_loss, symbol_params)
            if lr is None:
                lr = self.inner_lr
            fast_weights = [symbol_params[i] - lr * gvs[i] if gvs[i] is not None else symbol_params[i] for i in
                            range(len(gvs))]

            return fast_weights

        def get_loc(choice, all):
            choice_name = [i.name for i in choice]
            ind = []
            num = 0

            for i in all:
                if i.name in choice_name:
                    ind.append(num)
                num += 1
            return ind

        def rev_loc(choice, index, all):
            fill_choice = []
            index_pos = 0
            for ind, w in enumerate(all):
                if ind in index:
                    fill_choice.append(choice[index_pos])
                    index_pos += 1
                else:
                    fill_choice.append(w)
            return fill_choice

        def meta_task(input_tensor, inp, reuse=False, fwt=False, weights=None):
            if weights is None:
                weights = tf.trainable_variables()
            obs_vars, lab_vars, obs_q_vars, lab_q_vars = inp
            task_outputbs, task_lossesb, task_accs = [], [], []
            if self.model == 'rgcn':
                spt_output, emb, _ = forward_rgcn(input_tensor, obs_vars, weights, training_flag=self.training_tensor,
                                                  reuse=reuse, mp_num=self.mp_num, note='',
                                                  ss=self.module,neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
            else:
                spt_output, emb, _ = forward_han_sage_fwt(input_tensor, obs_vars, weights,
                                                      training_flag=self.training_tensor,
                                                      reuse=reuse, ss=self.module,mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num,
                                                      sa=self.sa, dropout=self.dropout, fwt=fwt)

            spt_loss = self.get_loss(spt_output, lab_vars)
            spt_acc = self.get_acc(spt_output, lab_vars)
            if self.task_mode is not None:
                attention = self.get_task_attention(emb, obs_vars, self.dropout)
            update_index = get_loc(self.model_params, weights)
            fast_weights = fast_params(spt_loss, self.model_params)
            fill_weights = rev_loc(fast_weights, update_index, weights)

            if self.model == 'rgcn':
                output_after, emb_p, _ = forward_rgcn(input_tensor, obs_q_vars, fill_weights, training_flag=self.training_tensor,
                                                      reuse=True, mp_num=self.mp_num, note='',
                                                      ss=self.module, neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
            else:
                output_after, emb_p, _ = forward_han_sage_fwt(input_tensor, obs_q_vars, fill_weights,
                                                      training_flag=self.training_tensor,
                                                      reuse=True, ss=self.module, mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num,
                                                      sa=self.sa, dropout=self.dropout, fwt=fwt)


            task_outputbs.append(output_after)
            task_lossesb.append(self.get_loss(output_after, lab_q_vars))

            for j in range(update_num - 1):

                if self.model == 'rgcn':
                    output_after, _, _ = forward_rgcn(input_tensor, obs_vars, fill_weights,
                                                      training_flag=self.training_tensor, reuse=True,
                                                      ss=self.module,mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
                else:
                    output_after, _, _ = forward_han_sage_fwt(input_tensor, obs_vars, fill_weights,
                                                              training_flag=self.training_tensor,
                                                              ss=self.module,reuse=True, mp_num=self.mp_num,
                                                              note='',
                                                              neighbor_num=self.neighbor_num,
                                                              sa=self.sa, dropout=self.dropout, fwt=fwt)

                loss_after = self.get_loss(output_after, lab_vars)
                fast_weights = fast_params(loss_after, fast_weights)
                fill_weights = rev_loc(fast_weights, update_index, fill_weights)

                if self.model == 'rgcn':
                    output_after, emb_p1, _ = forward_rgcn(input_tensor, obs_q_vars, fill_weights,
                                                      ss=self.module,training_flag=self.training_tensor, reuse=True,
                                                      mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
                else:
                    output_after, _, _ = forward_han_sage_fwt(input_tensor, obs_q_vars, fill_weights,
                                                              training_flag=self.training_tensor,
                                                              reuse=True, ss=self.module, mp_num=self.mp_num,
                                                              note='',
                                                              neighbor_num=self.neighbor_num,
                                                              sa=self.sa, dropout=self.dropout, fwt=fwt)

                task_outputbs.append(output_after)
                task_lossesb.append(self.get_loss(output_after, lab_q_vars))

            for j in range(update_num):
                task_accs.append(self.get_acc(task_outputbs[j], lab_q_vars))
            task_output = [spt_output, task_outputbs, spt_loss, task_lossesb, spt_acc, task_accs, []]
            if self.task_mode is not None:
                return task_output, attention
            else:
                return task_output

        def extract_batch_results(batch_results, meta_size):
            outputas = map(lambda x: x[0], batch_results)
            outputbs = map(lambda x: x[1], batch_results)
            lossesa = map(lambda x: x[2], batch_results) 
            lossesb = tf.concat(map(lambda x: tf.expand_dims(x[3], 1), batch_results), 1) 
            accuraciesa = map(lambda x: x[4], batch_results)
            accuraciesb = tf.concat(map(lambda x: tf.expand_dims(x[5], 1), batch_results), 1)

            total_loss1 = tf.reduce_sum(lossesa) / tf.cast(meta_size, self.stype)
            total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.cast(meta_size, self.stype) for j
                                                  in range(update_num)]
            total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.cast(meta_size, self.stype)
            total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.cast(meta_size, self.stype) for j in
                                      range(update_num)]
            return outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb, total_loss1, total_losses2, total_accuracy1, total_accuracies2, None

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            elems = batch_input(inp, meta_size)
            self.set_task_model()
            batch_results = []
            batch_tasks = []
            for i in range(meta_size):
                if self.task_mode is None:
                    result = meta_task(self.einput_tensor, elems[i], reuse=False, fwt=False)
                else:
                    result, emb_task = meta_task(self.einput_tensor, elems[i], reuse=False, fwt=False)
                    batch_tasks.append(emb_task)
                batch_results.append(result)

            _, _, _, _, _, _, self.eval_total_loss1, self.eval_total_losses2, self.eval_total_accuracy1, \
            self.eval_total_accuracies2,self.plot_embs = extract_batch_results(batch_results, meta_size)

    def forward_tri(self, ss_input1,ss_input2, ss_input3,  weights, reuse, cos=True):
	
        dropout = self.dropout
        drop = Dropout(dropout)
        act = tf.nn.relu
        def get_dis(ss_input, dom):
          
            node1 = tf.gather(ss_input, 0, axis=1)  
            node2 = tf.gather(ss_input, 1, axis=1)   

            if dom == 0 or dom ==1:
                if self.model == 'rgcn':
                    _, feat1, _ = forward_rgcn(self.sinput_tensor, node1, weights,
                                                training_flag=self.training_tensor, reuse=reuse,
                                                mp_num=self.mp_num, note='',
                                                ss=self.module,neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
                else:
                    _, feat1, _ = forward_han_sage_fwt(self.sinput_tensor, node1, weights, training_flag=self.training_tensor,
                                            reuse=reuse, ss=self.module,mp_num=self.mp_num,
                                            note='', neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout,
                                                fwt=False)
            else:
                if self.model == 'rgcn':
                    _, feat1, _ = forward_rgcn(self.tinput_tensor, node1, weights,
                                                training_flag=self.training_tensor, reuse=reuse,
                                                mp_num=self.mp_num, note='',
                                                ss=self.module,neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
                else:
                    _, feat1, _ = forward_han_sage_fwt(self.tinput_tensor, node1, weights, training_flag=self.training_tensor,
                                            reuse=reuse, ss=self.module,mp_num=self.mp_num,
                                            note='', neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout,
                                                fwt=False)

            if dom == 0:
                if self.model == 'rgcn':
                    _, feat2, _ = forward_rgcn(self.sinput_tensor, node2, weights,
                                                training_flag=self.training_tensor, reuse=reuse,
                                                mp_num=self.mp_num, ss=self.module, note='',
                                                neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
                else:

                    _, feat2, _ = forward_han_sage_fwt(self.sinput_tensor, node2, weights, training_flag=self.training_tensor,
                                                reuse=reuse, mp_num=self.mp_num,
                                                note='', neighbor_num=self.neighbor_num, sa=self.sa,
                                                dropout=self.dropout,
                                                fwt=False, ss=self.module)
            elif dom ==1 or dom ==2:
                if self.model == 'rgcn':
                    _, feat2, _ = forward_rgcn(self.tinput_tensor, node2, weights,
                                                training_flag=self.training_tensor, reuse=reuse,
                                                mp_num=self.mp_num, ss=self.module, note='',
                                                neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
                else:

                    _, feat2, _ = forward_han_sage_fwt(self.tinput_tensor, node2, weights, training_flag=self.training_tensor,
                                                reuse=reuse, mp_num=self.mp_num,
                                                note='', neighbor_num=self.neighbor_num, sa=self.sa,
                                                dropout=self.dropout,
                                                fwt=False, ss=self.module)
            

            ss_weights = [var for var in weights if 'aux' in var.name]
            
            def phi(feat, weights):
                ss_emb = tf.matmul(feat, weights[0])
                ss_emb = act(ss_emb + weights[1])
                ss_emb2 = tf.matmul(ss_emb, weights[2])
                ss_emb2 = ss_emb2 + weights[3]
                return ss_emb2

            if cos:  
                feat1 = drop(feat1, training=self.training_tensor)
                feat2 = drop(feat2, training=self.training_tensor)
                emb1 = phi(feat1, ss_weights)
                emb2 = phi(feat2, ss_weights)

                score_hat = emb1 * emb2
                score = tf.reduce_sum(score_hat, axis=1, keep_dims=True)
                return score
            else:
                feat = tf.concat(feat1, feat2)
                feat = drop(feat, training=self.training_tensor)
                logits = phi(feat, ss_weights)
                return logits
        pairp1 = get_dis(ss_input1, dom=0)
        pairp2 = get_dis(ss_input2, dom=2)
        pairn = get_dis(ss_input3, dom=1)
        loss1 = tf.math.maximum(0.0, 1.0 - pairp1 + pairn)
        loss2 = tf.math.maximum(0.0, 1.0 - pairp2 + pairn)
        return loss1+loss2


    def forward_mp(self, ss_input, weights, reuse, cos=True):
        dropout = self.dropout
        node1 = tf.gather(ss_input, 0, axis=1)  
        node2 = tf.gather(ss_input, 1, axis=1)   
        drop = Dropout(dropout)
        act = tf.nn.relu
 
        if self.model == 'rgcn':
            _, feat1, _ = forward_rgcn(self.sinput_tensor, node1, weights,
                                              training_flag=self.training_tensor, reuse=reuse,
                                              mp_num=self.mp_num, note='',
                                              ss=self.module,neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
        else:
            _, feat1, _ = forward_han_sage_fwt(self.sinput_tensor, node1, weights, training_flag=self.training_tensor,
                                           reuse=reuse, ss=self.module,mp_num=self.mp_num,
                                           note='', neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout,
                                               fwt=False)


        if self.model == 'rgcn':
            _, feat2, _ = forward_rgcn(self.sinput_tensor, node2, weights,
                                              training_flag=self.training_tensor, reuse=reuse,
                                              mp_num=self.mp_num, ss=self.module, note='',
                                              neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
        else:

            _, feat2, _ = forward_han_sage_fwt(self.sinput_tensor, node2, weights, training_flag=self.training_tensor,
                                               reuse=reuse, mp_num=self.mp_num,
                                               note='', neighbor_num=self.neighbor_num, sa=self.sa,
                                               dropout=self.dropout,
                                               fwt=False, ss=self.module)

        ss_weights = [var for var in weights if 'aux' in var.name]
        def phi(feat, weights):
            ss_emb = tf.matmul(feat, weights[0])
            ss_emb = act(ss_emb + weights[1])
            ss_emb2 = tf.matmul(ss_emb, weights[2])
            ss_emb2 = ss_emb2 + weights[3]
            return ss_emb2

        if cos:  
            feat1 = drop(feat1, training=self.training_tensor)
            feat2 = drop(feat2, training=self.training_tensor)
            emb1 = phi(feat1, ss_weights)
            emb2 = phi(feat2, ss_weights)

            score_hat = emb1 * emb2
            score = tf.reduce_sum(score_hat, axis=1, keep_dims=True)
            logits = score
        else:
            feat = tf.concat(feat1, feat2)
            feat = drop(feat, training=self.training_tensor)
            logits = phi(feat, ss_weights)
        return logits

    def opt(self, inp, inp2, meta_size, mode, update_num, ss_inp=None, tar_inp=None, super_setting=False,
            naive_setting=False, couple_setting=False):
        def batch_input(inp, meta_size):
            elems = []
            for i in range(meta_size):  
                elem = []
                elem.append(inp[i])
                elem.append(inp[1 * meta_size + i])  
                elem.append(inp[2 * meta_size + i])  
                elem.append(inp[3 * meta_size + i])  
                elems.append(elem)
            return elems

        def fast_params(symbol_loss, symbol_params, lr=None):
            gvs = tf.gradients(symbol_loss, symbol_params)
            if lr is None:
                lr = self.inner_lr
            fast_weights = [symbol_params[i] - lr * gvs[i] if gvs[i] is not None else symbol_params[i] for i in range(len(gvs))]
            return fast_weights

        def get_loc(choice, all):
            choice_name = [i.name for i in choice]
            ind = []
            num = 0

            for i in all:
                if i.name in choice_name:
                    ind.append(num)
                num += 1
            return ind

        def rev_loc(choice, index, all):
            fill_choice = []
            index_pos = 0
            for ind, w in enumerate(all):
                if ind in index:
                    fill_choice.append(choice[index_pos])
                    index_pos += 1
                else:
                    fill_choice.append(w)
            return fill_choice

        def meta_task(input_tensor, input_tensor2, inp, inp2, reuse=False, fwt=False, weights=None, modulate=False, reg_loss=False):

            obs_vars, lab_vars, obs_q_vars, lab_q_vars = inp

            task_outputbs, task_lossesb, task_accs = [], [], []

            if self.model == 'rgcn':
                spt_output, emb, _ = forward_rgcn(input_tensor, obs_vars, weights, training_flag=self.training_tensor,
                                                  ss=self.module,reuse=reuse, mp_num=self.mp_num, note='',
                                                  neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout,
                                                  modulate=modulate)
            else:
                spt_output, emb, _ = forward_han_sage_fwt(input_tensor, obs_vars, weights,
                                                      training_flag=self.training_tensor,
                                                      reuse=reuse, ss=self.module,mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num,
                                                      sa=self.sa, dropout=self.dropout, fwt=fwt)
            spt_loss = self.get_loss(spt_output, lab_vars)
            reg_loss = False
            if reg_loss is True:
                spt_loss += forward_reg(emb, weights, drop=self.dropout, training=self.training_tensor)
            spt_acc = self.get_acc(spt_output, lab_vars)

            update_index = get_loc(self.model_params, weights)
            fast_weights = fast_params(spt_loss, self.model_params)
            fill_weights = rev_loc(fast_weights, update_index, weights)

            if self.model == 'rgcn':
                output_after, emb_p, _ = forward_rgcn(input_tensor, obs_q_vars, fill_weights, training_flag=self.training_tensor,
                                                     ss=self.module, reuse=True, mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout)
            else:
                output_after, emb_p, _ = forward_han_sage_fwt(input_tensor, obs_q_vars, fill_weights,
                                                      training_flag=self.training_tensor,
                                                      reuse=True, ss=self.module, mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num,
                                                      sa=self.sa, dropout=self.dropout, fwt=fwt, modulate=modulate)

            task_outputbs.append(output_after)
            task_lossesb.append(self.get_loss(output_after, lab_q_vars))

            for j in range(update_num - 1):
                if self.model == 'rgcn':
                    output_after, emb_after, _ = forward_rgcn(input_tensor, obs_vars, fill_weights,
                                                      ss=self.module,training_flag=self.training_tensor, reuse=True,
                                                      mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout,
                                                      modulate=modulate)
                else:
                    output_after, emb_after, _ = forward_han_sage_fwt(input_tensor, obs_vars, fill_weights,
                                                              training_flag=self.training_tensor,
                                                              reuse=True, ss=self.module, mp_num=self.mp_num,
                                                              note='',
                                                              neighbor_num=self.neighbor_num,
                                                              sa=self.sa, dropout=self.dropout, fwt=fwt,modulate=modulate)
                loss_after = self.get_loss(output_after, lab_vars)

                reg_loss = False
                if reg_loss is True:
                    loss_after += forward_reg(emb_after, fill_weights, drop=self.dropout, training=self.training_tensor)
                fast_weights = fast_params(loss_after, fast_weights)
                fill_weights = rev_loc(fast_weights, update_index, fill_weights)
                if self.model == 'rgcn':
                    output_after, _, _ = forward_rgcn(input_tensor, obs_q_vars, fill_weights,
                                                      ss=self.module,training_flag=self.training_tensor, reuse=True,
                                                      mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout,
                                                      modulate=modulate)
                else:
                    output_after, _, _ = forward_han_sage_fwt(input_tensor, obs_q_vars, fill_weights,
                                                              training_flag=self.training_tensor,
                                                              reuse=True, ss=self.module, mp_num=self.mp_num,
                                                              note='',
                                                              neighbor_num=self.neighbor_num,
                                                              sa=self.sa, dropout=self.dropout, fwt=fwt,modulate=modulate)

                task_outputbs.append(output_after)
                loss_final = self.get_loss(output_after, lab_q_vars)

                task_lossesb.append(loss_final)

            reg_loss = True
            if reg_loss is True:
                obs_vars2, lab_vars2, obs_q_vars2, lab_q_vars2 = inp2
                d1out, d1emb, _ = forward_rgcn(input_tensor, obs_vars, fill_weights,
                                                          ss=self.module,training_flag=self.training_tensor, reuse=True,
                                                          mp_num=self.mp_num, note='',
                                                          neighbor_num=self.neighbor_num, sa=self.sa,
                                                          dropout=self.dropout,
                                                          modulate=modulate)
                
                d2out, d2emb, _ = forward_rgcn(input_tensor2, obs_vars2, fill_weights,
                                                          training_flag=self.training_tensor, reuse=True,
                                                          mp_num=self.mp_num, ss=self.module, note='',
                                                          neighbor_num=self.neighbor_num, sa=self.sa,
                                                          dropout=self.dropout,
                                                          modulate=modulate)
                reg_loss1 = forward_reg(d1emb, fill_weights, drop=self.dropout, training=self.training_tensor)
                reg_loss2 = forward_reg(d2emb, fill_weights, drop=self.dropout, training=self.training_tensor)
                reg_loss = tf.norm(reg_loss1-reg_loss2, ord='euclidean')
		
                update_index_f = get_loc(self.fea_params, weights)

                fast_weights_s = fast_params(reg_loss, fast_weights[:-2], self.inner_lr)
                fill_weights_s = rev_loc(fast_weights_s, update_index_f, weights)
		
                output_after2, emb2, _ = forward_rgcn(input_tensor2, obs_vars2, fill_weights_s,
                                                      ss=self.module,training_flag=self.training_tensor, reuse=True,
                                                      mp_num=self.mp_num, note='',
                                                      neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout,
                                                      modulate=modulate)
		
                update_index2 = get_loc(self.cls_params, weights)
                cls_adapt = self.get_loss(output_after2, lab_vars2)
                fast_weights2 = fast_params(cls_adapt, weights[-2:], self.inner_lr)
                fill_weights2 = rev_loc(fast_weights2, update_index2, fill_weights_s)


                output_after2, _, _ = forward_rgcn(input_tensor2, obs_q_vars2, fill_weights2,
                                                   ss=self.module,training_flag=self.training_tensor, reuse=True,
                                                   mp_num=self.mp_num, note='',
                                                   neighbor_num=self.neighbor_num, sa=self.sa, dropout=self.dropout,
                                                   modulate=modulate)
                with tf.control_dependencies([]):
                    task_losses2 = self.get_loss(output_after2, lab_q_vars2)

            for j in range(update_num):
                task_accs.append(self.get_acc(task_outputbs[j], lab_q_vars))
            task_output = [spt_output, task_outputbs, spt_loss, task_lossesb, spt_acc, task_accs, None]
            return task_output, task_losses2

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            if mode == 'train':
                elems = batch_input(inp, meta_size)
                elems2 = batch_input(inp2, meta_size)
                self.set_task_model()
                meta_weights = tf.trainable_variables()
                self.global_step = tf.Variable(0, trainable=False, name='set_global_step')
            
                warm_step = tf.constant(2000, dtype=tf.int32)
                    
                learning_rate = tf.train.exponential_decay(self.lr, self.global_step, self.decay_steps, self.decay_rate, staircase=False)
                optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-7)
                reg_optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-7)
		        
                if couple_setting is True:
                    def couple_task(i, domain, elem, elem2, weights, modulate):
                        inputs = self.sinput_tensor
                        inputs2 = self.tinput_tensor
                        if weights is None:
                            weights = meta_weights
                        batch_results = []
                        assert self.task_mode is None
                        result, tdloss = meta_task(inputs, inputs2, elem, elem2, reg_loss=True, reuse=False, fwt=False,
                                                    weights=weights, modulate=modulate)
                        batch_results.append(result)
                        lossesb = tf.concat(map(lambda x: tf.expand_dims(x[3], 1), batch_results),
                                            1) 
                        model_loss = lossesb[-1]
                        pembs = map(lambda x: x[-1], batch_results)
                        return model_loss, tdloss, pembs

                    def warmuploss(lo1, lo2, gstep, wstep):
                        wloss = tf.cond(pred = tf.less(gstep, wstep),
                        true_fn = lambda: lo2,
                        false_fn = lambda: lo1 + lo2)
                        return wloss

                    def adaption():
                        couple_loss1 = []
                        tdloss1 = []
                        bzpembs = []
                        for i in range(meta_size):
                            model_loss, tdloss, pemb = couple_task(i, 's', elems[i], elems2[i], None, modulate=False)
                            couple_loss1.append(model_loss)
                            tdloss1.append(tdloss)
                            bzpembs.append(pemb)
                        loss1 = tf.reduce_mean(couple_loss1)
                        loss2 = 0.1 * tf.reduce_mean(tdloss1)

                        with tf.control_dependencies([]):
                            loss = loss1 + loss2 + self.tri_loss(ss_inp, meta_weights)
                            couple_gvs1 = optimizer.compute_gradients(loss, self.model_params)
                            couple_gvs1 = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in couple_gvs1 if
                                        grad is not None]
                            couple_op1 = optimizer.apply_gradients(couple_gvs1, global_step=self.global_step)

                        with tf.control_dependencies([]):
                            couple_gvs2 = reg_optimizer.compute_gradients(loss2, self.reg_params)
                            with tf.control_dependencies([]):
                                couple_gvs2 = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in couple_gvs2 if
                                           grad is not None]
                            couple_op2 = reg_optimizer.apply_gradients(couple_gvs2)
                            self.current_lr1 = reg_optimizer._lr
                            self.current_lr2 = optimizer._lr
                    
                            couple_op = tf.cond(pred=tf.less(self.global_step, warm_step),
                                true_fn=lambda: tf.group(couple_op1, couple_op2),
                                false_fn = lambda: couple_op1)
               
                            return couple_op
                    self.adaption = adaption()

    def make_vars(self, batch, stepnum='sup'):
        obs_vars, lab_vars = [], []
        for i in range(batch):
            obs_vars.append(tf.placeholder(dtype=self.inttype, shape=(None,),
                                           name=stepnum + '_' + str(i)))
            lab_vars.append(
                tf.placeholder(dtype=self.inttype, shape=(None, self.mapped_dim_y), name= stepnum + '_l' + str(i)))
        return obs_vars, lab_vars

    def tri_loss(self, ss_inp=None, weights=None):
        ss_input1, ss_input2, ss_input3 = ss_inp
        ss_loss  = self.forward_tri(ss_input1, ss_input2, ss_input3,  weights, reuse=False)
        lvars = tf.trainable_variables()
        aux_loss = self.ss_alpha * ss_loss
        return aux_loss
    
    def aux_loss(self, ss_inp=None, weights=None):
        ss_input, ss_label = ss_inp
        ss_cls = self.forward_mp(ss_input, weights, reuse=False)

        lvars = tf.trainable_variables()
        lossl2 = tf.add_n([tf.nn.l2_loss(v) for v in lvars if 'bias' not in v.name])
        ss_loss = self.get_mlloss(ss_cls, ss_label)

        decay_steps = tf.constant(500)
        decay_rate = tf.constant(0.95)

        alpha = self.ss_alpha * tf.pow(decay_rate,
                                            tf.cast(self.global_step, self.stype) / tf.cast(decay_steps,
                                                                                            self.stype))
        aux_loss = alpha * ss_loss  
        return aux_loss

    def meta_loss(self, lossesb, update_num, ss_inp=None, batch_tasks=None):
        if self.task_mode == 'mean':
            batch_mean = tf.squeeze(tf.stack(batch_tasks, axis=0))
            batch_m = tf.reduce_mean(batch_mean, axis=0, keepdims=True)
            ee = K.dot(batch_m, K.permute_dimensions(batch_mean, [1, 0]))
            ee = K.softmax(ee)
            weighted_batch_loss = ee * lossesb[update_num - 1]
            return weighted_batch_loss
        elif (self.task_mode == 'emb_mat') or (self.task_mode == 'emb_pooling') or (
                self.task_mode == 'att_residual') or (self.task_mode == 'degree'):

            ss_input, ss_label = ss_inp
            ss_cls = self.forward_mp(ss_input, self.weights, reuse=False, dropout=self.dropout)
            ss_loss = self.get_mlloss(ss_cls, ss_label)
            decay_steps = tf.constant(500)
            decay_rate = tf.constant(0.95)
            with tf.control_dependencies([]):
                alpha = self.ss_alpha * tf.pow(decay_rate,
                                                tf.cast(self.global_step, self.stype) / tf.cast(decay_steps,
                                                                                                self.stype))

            lvars = tf.trainable_variables()
            lossl2 = tf.add_n([tf.nn.l2_loss(v) for v in lvars if 'bias' not in v.name])
            losses = lossesb[update_num - 1]
            code_sum = tf.reduce_sum(batch_tasks)
            batch_tasks = [bt / code_sum for bt in batch_tasks]
            weighted_loss = [batch_tasks[i] * lossesb[update_num - 1][i] for i in range(len(batch_tasks))]
            loss = tf.Print(lossesb[update_num - 1], [lossesb[update_num - 1]], message='loss')

            all_loss = tf.reduce_sum(weighted_loss) + self.ss_alpha * ss_loss
            
            return all_loss
        else:
            ss_input, ss_label = ss_inp
            ss_cls = self.forward_mp(ss_input, self.weights, reuse=False, dropout=self.dropout)
            lvars = tf.trainable_variables()
            lossl2 = tf.add_n([tf.nn.l2_loss(v) for v in lvars if 'bias' not in v.name])

            with tf.control_dependencies([]):
                ss_loss = self.get_mlloss(ss_cls, ss_label)

            loss = self.meta_total_losses2[update_num - 1]
            decay_steps = tf.constant(500)
            decay_rate = tf.constant(0.95)

            with tf.control_dependencies([]):
                alpha = self.ss_alpha * tf.pow(decay_rate,
                                                tf.cast(self.global_step, self.stype) / tf.cast(decay_steps,
                                                                                                self.stype))
                aux_loss = loss + alpha * ss_loss   

            return aux_loss
        