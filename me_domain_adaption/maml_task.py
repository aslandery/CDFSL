from utils import *
from tensorflow.python.keras import backend as K
from tensorflow.contrib.layers import layer_norm
from tensorflow.python.keras.layers import Dropout



class BaseTask(object):
    def set_task_model(self):
        if self.impo == 'degree':
            self.get_score_d()
        elif self.impo == 'embd':
            self.get_score_embd()
        elif self.impo == 'vec+':
            self.get_score_vec()


    def get_score_d(self):
        impo = 16
        self.degree_input = tf.placeholder(dtype=self.stype, shape=(None, self.mp_num),
                                           name='degree')
        self.impo_w2 = tf.get_variable(name='impo_w2', shape=(self.mp_num, impo), dtype=self.stype,
                                       initializer=tf.glorot_uniform_initializer, trainable=True)
        self.impo_b2 = tf.get_variable(name='impo_bias2', shape=(impo,), trainable=True, dtype=self.stype)
        self.impo_w3 = tf.get_variable(name='impo_w3', shape=(impo, 1), dtype=self.stype,
                                       initializer=tf.glorot_uniform_initializer, trainable=True)
        self.impo_b3 = tf.get_variable(name='impo_bias3', shape=(1,), trainable=True, dtype=self.stype)


    def get_score_vec(self):
        vec = 64
        self.emb_trans = tf.get_variable(name='emb_trans', shape=(self.hdims[-1], vec), dtype=self.stype,
                                         initializer=tf.glorot_uniform_initializer, trainable=True)
        self.att_vector = tf.get_variable(name='att_vec', shape=(vec, 1), dtype=self.stype,
                                          initializer=tf.glorot_uniform_initializer, trainable=True)
        self.degree_input = tf.placeholder(dtype=self.stype, shape=(None, self.mp_num),
                                           name='degree')
        self.ret_weight = tf.get_variable(name='impo_w2', shape=(vec, 1), dtype=self.stype,
                                          initializer=tf.glorot_uniform_initializer, trainable=True)
        self.ret_bias = tf.get_variable(name='impo_bias2', shape=(1,), trainable=True, dtype=self.stype)


    def get_score_embd(self):
        self.degree_input = tf.placeholder(dtype=self.stype, shape=(None, self.mp_num),
                                           name='degree')
        self.can_weight = tf.get_variable(name='impo_w2', shape=(self.mp_num + self.hdims[-1], 1), dtype=self.stype,
                                          initializer=tf.glorot_uniform_initializer, trainable=True)
        self.can_bias = tf.get_variable(name='impo_bias2', shape=(1,), trainable=True, dtype=self.stype)


    def get_task_model(self, input_dim):
        with tf.variable_scope('task_model', reuse=tf.AUTO_REUSE):
            self.task_mat = tf.get_variable(name='task_mat', shape=(input_dim, 1), dtype=self.stype,
                                            initializer=tf.glorot_uniform_initializer, trainable=True)
            self.task_bias = tf.get_variable(name='task_bias', shape=(1,), trainable=True, dtype=self.stype)
            self.task_weight = tf.get_variable(name='task_w', shape=(1, self.nway * self.kshot), dtype=self.stype,
                                               initializer=tf.glorot_uniform_initializer, trainable=True)


    def get_attention(self, in_dim, output_dim):
        with tf.variable_scope('att_model', reuse=tf.AUTO_REUSE):
            self.att_mat = tf.get_variable(name='attention_mat', shape=(in_dim, output_dim), dtype=self.stype,
                                           initializer=tf.glorot_uniform_initializer, trainable=True)
            return output_dim


    def get_task_attention(self, emb, ind, dropout):
        def impo(self, emb, ind, dropout):
            if self.impo == 'degree':
                score = self.impo_d(self, emb, ind, dropout=dropout)
            elif self.impo == 'embd':
                score = self.impo_ed(self, emb, ind, dropout=dropout)
            elif self.impo == 'vec+':
                emb_ = tf.matmul(emb, self.emb_trans) 
                score = tf.matmul(emb_, self.att_vector)  
                score = tf.reshape(score, [1, -1])  
                score_nor = tf.nn.softmax(score)  
                ret = tf.matmul(score_nor, emb_)
                score = tf.nn.sigmoid(tf.matmul(ret, self.ret_weight) + self.ret_bias)
            return score

        if self.task_mode == 'mean':
            emb = tf.reshape(tf.reduce_mean(emb, axis=0), [1, -1])
        elif self.task_mode == 'emb_mat':
            emb = Dropout(self.dropout)(emb, training=self.training_tensor)
            emb_mat = tf.matmul(emb, self.task_mat) + self.task_bias  
            with tf.control_dependencies([]):
                emb_s = tf.matmul(self.task_weight, emb_mat)  
                emb = tf.nn.sigmoid(emb_s)
        elif self.task_mode == 'emb_pooling':
            emb = tf.nn.sigmoid(
                tf.matmul(tf.reduce_mean(emb, axis=0, keepdims=True), self.task_mat) + self.task_bias)
        elif self.task_mode == 'att_residual':
            emb = self.emb_attention(self, emb, pos=True)
        elif self.task_mode == 'degree':
            emb = tf.reduce_mean(impo(self, emb, ind, dropout))
        return emb


    def impo_d(self, emb, ind, dropout):
        act = tf.nn.relu
        drop = Dropout(dropout)
        degree = tf.log(tf.gather(self.degree_input, ind) + 0.5)  
        degree = drop(degree, training=self.training_tensor)
        degree_score = act(tf.matmul(degree, self.impo_w2) + self.impo_b2)  

        mscore = tf.nn.sigmoid(tf.matmul(degree_score, self.impo_w3) + self.impo_b3)
        return mscore


    def impo_ed(self, emb, ind, dropout):
        degree = tf.log(tf.gather(self.degree_input, ind) + 0.5)  
        drop = Dropout(dropout)
        emb = tf.concat([emb, degree], axis=1)
        emb_ = drop(emb, training=self.training_tensor)
        score = tf.nn.sigmoid(tf.matmul(emb, self.can_weight) + self.can_bias)
        return score


    def emb_attention(self, emb, pos=False): 
        if pos:
            ee = K.dot(K.dot(emb, self.att_mat), tf.transpose(emb))
            with tf.control_dependencies([]):
                ee = K.softmax(ee)
            with tf.control_dependencies([]):
                oo = tf.nn.relu(K.dot(ee, emb))
            emb_con = tf.concat((emb, oo), axis=-1)
            emb_add = tf.math.add(emb, oo)
            emb_add_ln = layer_norm(emb_add)
            emb_att_mat = tf.reduce_mean(tf.nn.sigmoid((tf.matmul(emb_add_ln, self.task_mat) + self.task_bias)),
                                         axis=0) 
        else:
            emb_ln = layer_norm(emb)
            ee_ln = K.dot(K.dot(emb_ln, self.att_mat), tf.transpose(emb))
            ee_ln = K.softmax(ee_ln)
            oo_ln = tf.nn.relu(K.dot(ee_ln, emb_ln))
            emb_add = tf.math.add(emb, oo_ln)
            emb_att_mat = tf.reduce_mean(tf.nn.sigmoid((tf.matmul(emb_add, self.task_mat) + self.task_bias)),
                                         axis=0) 
        return emb_att_mat