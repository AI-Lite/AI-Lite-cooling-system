#coding:utf-8
import tensorflow.compat.v1 as tf
import numpy as np

class Graph_1:

    def __init__(self,LR_1,Input_dim_1,Num_hidden_1,Out_dim_1):
        self.g1 = tf.Graph()
        #self.ac_num = Ac_num
        #self.rc_num = Rc_num
        self.input_dim_1 = Input_dim_1  # the dimension of the input of the graph1
        self.num_hidden_1 = Num_hidden_1  # the number of dimension of graph1
        self.out_dim_1 = Out_dim_1  # the number of output of graph1
        self.lr_1 = LR_1  # the step_size(learning rate) of graph1

        with self.g1.as_default():  # definition of graph1
        #def build_net(self):
            with tf.name_scope('inputs'):
                self.observation_1 = tf.placeholder(tf.float32, [None, self.input_dim_1], name="observation_1")
                self.action_1 = tf.placeholder(tf.int32, [None,], name="action_1")
                self.reward_1 = tf.placeholder(tf.float32, [None,], name="reward_1")
                self.lrmult_1 = tf.placeholder(tf.float32, (None), name='lrmult_1')
            layer = tf.layers.dense(  # 全连接层，输入为观测值，单元个数10，
                inputs=self.observation_1,
                units=self.num_hidden_1,
                #activation=tf.nn.relu,  # tanh activation
                #activation = tf.nn.sigmoid,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                # kernel_initializer=tf.constant_initializer(0.1),
                #kernel_initializer=tf.random_normal_initializer(mean=0.2, stddev=0.1),  # 返回一个生成具有正态分布的张量的初始化器，权重矩阵的初始化函数
                bias_initializer=tf.constant_initializer(0.1),
                name='fc1'
            )
            # fc2
            all_act = tf.layers.dense(  # 输入为上一级输出layer，单元数为动作个数
                inputs=layer,
                #inputs=self.observation_1,
                units=self.out_dim_1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
                # kernel_initializer=tf.constant_initializer(-0.02),
                #kernel_initializer=tf.constant_initializer(0),
                #kernel_initializer=tf.random_normal_initializer(mean=0.1, stddev=0.09),  # 返回一个生成具有正态分布的张量的初始化器
                bias_initializer=tf.constant_initializer(0.1),
                name='fc2'
            )
            self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # 加一个softmax回归，将动作值转化为概率值

            with tf.name_scope('loss_1'):
                cross_entropy_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                                 labels=self.action_1)
                loss_1 = tf.reduce_mean((cross_entropy_1) * (self.reward_1))
            with tf.name_scope('train_1'):
                #opt = tf.train.AdamOptimizer(self.lr_1)
                #gradients = opt.compute_gradients(loss_1)
                #capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                #self.train_op_1 = opt.apply_gradients(capped_gradients)
                self.train_op_1 = tf.train.AdamOptimizer(self.lr_1 * self.lrmult_1).minimize(loss_1)
                self.init_op1 = tf.global_variables_initializer()
                self.saver_1 = tf.train.Saver(max_to_keep = 100)
                # self.reset = tf.reset_default_graph()
        self.sess1 = tf.Session(graph=self.g1)


class Graph_2:
    def __init__(self,LR_2,Input_dim_2,Num_hidden_2,Out_dim_2):
        self.g2 = tf.Graph()
        #self.ac_num = Ac_num
        #self.rc_num = Rc_num
        self.input_dim_2 = Input_dim_2  # the dimension of the input of the graph2
        self.num_hidden_2 = Num_hidden_2  # the number of dimension of graph2
        self.out_dim_2 = Out_dim_2  # the number of output of graph2
        self.lr_2 = LR_2  # the step_size(learning rate) of graph2

        with self.g2.as_default():  # definition of graph1
            with tf.name_scope('inputs'):
                self.observation_2 = tf.placeholder(tf.float32, [None, self.input_dim_2], name="observation_1")
                self.action_2 = tf.placeholder(tf.int32, [None,], name="action_2")
                self.reward_2 = tf.placeholder(tf.float32, [None], name="reward_2")
                self.lrmult_2 = tf.placeholder(tf.float32, (None), name='lrmult_2')
            layer = tf.layers.dense(  # 全连接层，输入为观测值，单元个数10，
                inputs=self.observation_2,
                units=self.num_hidden_2,
                activation=tf.nn.relu,  # tanh activation
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),  # 返回一个生成具有正态分布的张量的初始化器，权重矩阵的初始化函数
                bias_initializer=tf.constant_initializer(0.1),
                name='fc1'
            )
            # fc2
            all_act = tf.layers.dense(  # 输入为上一级输出layer，单元数为动作个数
                inputs=layer,
                units=self.out_dim_2,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),  # 返回一个生成具有正态分布的张量的初始化器
                bias_initializer=tf.constant_initializer(0.1),
                name='fc2'
            )
            self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # 加一个softmax回归，将动作值转化为概率值

            with tf.name_scope('loss_2'):
                cross_entropy_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                                 labels=self.action_2)
                loss_2 = tf.reduce_mean((cross_entropy_2) * (self.reward_2))
            with tf.name_scope('train_2'):
                #opt = tf.train.AdamOptimizer(self.lr_2)
                #gradients = opt.compute_gradients(loss_2)
                #capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
                #self.train_op_2 = opt.apply_gradients(capped_gradients)
                self.train_op_2 = tf.train.AdamOptimizer(self.lr_2 * self.lrmult_2).minimize(loss_2)
                self.init_op2 = tf.global_variables_initializer()
                self.saver_2 = tf.train.Saver(max_to_keep = 200)
        self.sess2 = tf.Session(graph=self.g2)