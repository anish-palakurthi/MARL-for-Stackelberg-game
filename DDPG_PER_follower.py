import time
import numpy as np
import tensorflow as tf

SEED = int(time.clock()*100000000%100)
np.random.seed(SEED)
tf.set_random_seed(SEED)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0, 即TD误差
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


LR_A = 0.001    # learning rate for actor
LR_C = 0.005    # learning rate for critic
GAMMA = 0.95     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128

from MATest.State_Transition_AutoRegressive_AF import power_relay_max, cost_limitation_per_watt, power_source_max, alpha, relay_num
A_BOUND = [1.0]
S_DIM = 2*relay_num+1 + 2
A_DIM = 1


class DDPG_PER(object):
    def __init__(self, a_dim=A_DIM, s_dim=S_DIM, a_bound=A_BOUND):
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            self.q_ = self._build_c(self.S_, self.a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e) for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        self.q_target = self.R + GAMMA * self.q_
        self.abs_errors = tf.abs(self.q_target - self.q)  # for updating Sumtree
        self.td_error = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q))

        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)  # Critic 优化 TD-Error

        self.a_loss = - tf.reduce_mean(self.q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(self.a_loss, var_list=self.ae_params)    # Actor 优化动作概率

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

    def learn(self, batchsize=64):
        self.sess.run(self.soft_replace)  # 软更新

        tree_idx, bt, ISWeights = self.memory.sample(n=batchsize)  # PER

        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain,
                      feed_dict={self.S: bs, self.ISWeights: ISWeights}
                      )
        _, abs_errors = self.sess.run([self.ctrain, self.abs_errors],
                                      feed_dict={self.S: bs, self.a: ba, self.R: br, self.S_: bs_,
                                                 self.ISWeights: ISWeights})
        self.memory.batch_update(tree_idx, abs_errors)  # update priority

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        self.memory.store(transition)  # have high priority for newly arrived transition
        if self.pointer < MEMORY_CAPACITY:
            self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net_1 = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)
            net_2 = tf.layers.dense(net_1, 64, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net_2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_2 = tf.layers.dense(net_1, 64, activation=tf.nn.relu, name='l2', trainable=trainable)
            return tf.layers.dense(net_2, 1, trainable=trainable)  # Q(s,a)

