# encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
# ---------------------------------------------------------------------------------------------
# TEMN model (Zhou, Mascolo, Zhao, KDD 2019), adapted from https://github.com/XiaoZHOUCAM/TEMN
#
# What this file builds (one TF1-style static graph, trained with sessions and placeholders):
#   Memory network  (paper 4.2)  q_v = learned vector per POI id; p_u = mean of q over the user's
#                                history (Eq. 2); key e_uv = p_u * q_v (Eq. 3); attention over the
#                                memory keys (Eq. 4); relation r_uv = weighted sum of memory slots
#                                (Eq. 5); score s_m = -||p_u + r_uv - q_v|| (Eq. 6); hinge loss (Eq. 7)
#   TLDA link       (paper 4.4)  softmax(W p_u + b) must match the user's TLDA topic mix (Eq. 12-14)
#   Geography       (paper 4.5)  s_geo = (rho_u + rho_v) * d(c_u, v) + b_u (Eq. 17); hinge loss (Eq. 18)
#   Joint loss      (paper 4.6)  L = L_m + ratio1 * L_topic + ratio2 * L_geo + l2 (Eq. 20)
#   Ranking score   (paper 4.6)  s = s_m + eta * s_geo (Eq. 19)
#
# Comment markers used in src/ (full list in reports/STATUS_AND_NEXT_STEPS.md):
#   [CHANGED Cn]  this line differs from the authors' code
#   [NOTE Nn]     authors' code kept as it is, but worth knowing (a quirk or a deviation from the paper)
# ---------------------------------------------------------------------------------------------
from __future__ import division
# [CHANGED C1] upstream: `import tensorflow as tf` (TF 1.12). The TF1 graph/session API is kept
# and runs on TF 2.x through the compat module, so the model itself is unchanged.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# [CHANGED C4] upstream: `from basemodel import basemodel`; now imported as a package from the repo root
from src.basemodel import basemodel


class TEMN(basemodel):
    """
    @num_users:number of users
    @num_items:number of POIs
    @lamb_m:the margin of memory network
    @lamb_d:the margin of dis_model
    @ratio1:the influence of topic_model
    @ratio2:the influence of dis_model
    """

    def __init__(self, num_users, num_items, args, sem_table=None):
        print('creating my TEMN!')
        self.num_users = num_users
        self.num_items = num_items
        self.graph = tf.Graph()
        self.args = args
        self.stddev = self.args.stddev
        self.learn_rate = self.args.learn_rate
        self.lamb_m = args.lamb_m
        self.lamb_d = args.lamb_d
        self.ratio1 = args.ratio1
        self.ratio2 = args.ratio2
        self.attention = None
        self.selected_memory = None
        self.num_mem = self.args.num_mem
        self.sem_table = sem_table

        self.initializer = self._get_initializer()
        self._set_opt()
        self._creat_model_inputs()
        self._build_list_network()

    def get_list_feed_dict(self, batch, mode='training'):
        # Turns a batch of tuples (built by process_TEMN_data) into placeholder values.
        #   training tuple: (u, positive POI, dist(u, pos), history, negative POI, TLDA topics of u, dist(u, neg))
        #   test tuple:     (u, candidate POI, dist(u, cand), history); a test group is 1 positive + negatives
        # history = the user's distinct training POIs (in training, minus the positive itself).
        def process_all_items(x_all_item):
            # Pads the history with id 0 up to max_p_num; the padding is masked out below.
            # [NOTE N6] id 0 is also a real POI, so the padding is a copy of POI 0's vector until masked.
            # [NOTE N5] a longer history keeps its FIRST max_p_num items. The history comes from a
            # Python set, so that is an arbitrary subset (in practice the lowest POI ids), not the
            # most recent POIs. Affects users with > max_p_num training POIs (at 200: 6 NYC, 22 Tokyo users).
            cur_all_ii = [0 for j in range(self.args.max_p_num)]
            for j in range(len(x_all_item)):
                if j < self.args.max_p_num:
                    cur_all_ii[j] = x_all_item[j]
            return cur_all_ii

        if mode == 'training':
            user_input = [x[0] for x in batch]
            item_input = [x[1] for x in batch]
            uindist = [x[2] for x in batch]
            all_items_data = [process_all_items(x[3]) for x in batch]
            # [CHANGED C6] upstream: ll = [len(x[3]) for x in batch]
            # ll is the divisor of the mean p_u (Eq. 2). The history is cut to max_p_num items above,
            # so the count must be cut too; upstream divided a sum of max_p_num vectors by the full
            # length (e.g. 200 vectors by 300) and shrank p_u for every user with a long history.
            ll = [min(len(x[3]), self.args.max_p_num) for x in batch]
            item_input_neg = [x[4] for x in batch]
            topic_input = [x[5] for x in batch]
            uindistneg = [x[6] for x in batch]
            feed_dict = {
                self.user_input: user_input,  # user id
                self.item_input: item_input,  # item id
                self.item_input_neg: item_input_neg,
                self.L: ll,  # number of POIs a user visited
                self.all_items: all_items_data,  # POI list a user has visited
                self.label: topic_input,  # user-topic from TLDA
                self.DIST: uindist,  # distance from user to positive POI
                self.DIST_neg: uindistneg,  # distance from user to negative POI
                # [NOTE N4] fed, but no op in the graph uses self.dropout: TEMN has no dropout.
                # (Do not confuse it with the q_v dropout p of Semantic TEMN, which is still to be added.)
                self.dropout: self.args.dropout
            }
            # [CHANGED C23] ID dropout mu, fed in training only; test keeps the default 0 (no dropout)
            if self.sem_table is not None:
                feed_dict[self.id_drop] = self.args.id_drop
        else:
            # test mode: no negative, no topics. Every candidate of a group goes through item_input.
            user_input = [x[0] for x in batch]
            item_input = [x[1] for x in batch]
            uindist = [x[2] for x in batch]
            all_items_data = [process_all_items(x[3]) for x in batch]
            # [CHANGED C6] upstream: ll = [len(x[3]) for x in batch] (see the training branch)
            ll = [min(len(x[3]), self.args.max_p_num) for x in batch]
            feed_dict = {
                self.user_input: user_input,
                self.item_input: item_input,
                self.L: ll,
                self.DIST: uindist,
                self.all_items: all_items_data,
                self.dropout: 1
            }
        feed_dict[self.learn_rate] = self.args.learn_rate
        return feed_dict

    def _creat_model_inputs(self):
        self.user_input = tf.placeholder(tf.int32, shape=[None], name='user')
        self.item_input = tf.placeholder(tf.int32, shape=[None], name='item')
        self.item_input_neg = tf.placeholder(tf.int32, shape=[None], name='item_neg')
        self.input_type = tf.placeholder(tf.int32, shape=[None], name='type')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.label = tf.placeholder(tf.float32, shape=[None, self.args.topic_num], name='labels')

        self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        self.L = tf.placeholder(tf.float32, shape=[None], name='L')
        self.DIST = tf.placeholder(tf.float32, shape=[None], name='DIST')
        self.DIST_neg = tf.placeholder(tf.float32, shape=[None], name='DIST_neg')
        self.all_items = tf.placeholder(tf.int32, shape=[None, self.args.max_p_num], name="HISTORY")
        self.batch_size = tf.shape(self.item_input)[0]

    def _composition_layer(self, user_emb, item_emb, dist='L2', selected_memory=None):
        # Memory score s_m (Eq. 6): minus the distance between q_v and p_u + r_uv (TransE-style).
        # [NOTE N3] the paper squares the distance (||.||_2^2); the code uses sqrt(sum of squares + 1e-3),
        # i.e. the distance itself. Same order of POIs for s_m alone, but a different scale once
        # eta * s_geo is added (Eq. 19) and inside the hinge loss.
        energy = item_emb - (user_emb + selected_memory)
        if 'L2' in dist:
            final_layer = -tf.sqrt(tf.reduce_sum(tf.square(energy), 1) + 1E-3)
        elif 'L1' in dist:
            final_layer = -tf.reduce_sum(tf.abs(energy), 1)
        else:
            raise Exception('Please specify distance metric')
        final_layer = tf.reshape(final_layer, [-1, 1])
        return final_layer

    def _get_prediction(self, user_emb, item_emb, memory_key):
        # [NOTE N4] dead code in the authors' version: never called, and self.memory_value does not exist.
        # The same computation is written inline in _build_list_network.
        _key = tf.multiply(self.user_emb, self.item_emb)
        _key = tf.expand_dims(_key, 1)
        key_attention = tf.squeeze(tf.matmul(_key, memory_key))
        key_attention = tf.nn.softmax(key_attention)
        selected_memory = tf.matmul(key_attention, self.memory_value)
        final_layer = self._composition_layer(user_emb, item_emb, selected_memory=selected_memory)
        return final_layer

    def _build_list_network(self):
        stddev = self.stddev
        with tf.variable_scope('embedding_layer', initializer=self.initializer):
            with tf.device('/cpu:0'):
                # ---- Memory network parameters (paper 4.2.2): keys K [d x h] and memory slots M [h x d]
                self.user_item_key = tf.Variable(
                    tf.random_normal(
                        [self.args.embedding_size, self.num_mem],
                        stddev=stddev))
                self.memories = tf.Variable(
                    tf.random_normal(
                        [self.num_mem, self.args.embedding_size],
                        stddev=stddev))

                # ---- POI embeddings q_v (paper 4.2.1): one learned row per POI id.
                # This is the table Semantic TEMN changes to  q~_v = m_v * q_v + W e_v  (proposal Eq. 2).
                self.item_embeddings = tf.get_variable('item_emb', [self.num_items + 1, self.args.embedding_size],
                                                       initializer=self.initializer)
                # [CHANGED C23] Semantic TEMN (proposal Eq. 2): q~_v = m_v * q_v + W e_v for every POI row.
                # Without a semantic table no op is added, so the graph is exactly base TEMN.
                poi_vectors = self.item_embeddings
                if self.sem_table is not None:
                    # e_v per POI (frozen), row = poi_id; keep the model's num_items + 1 rows
                    sem_table = tf.constant(self.sem_table[:self.num_items + 1], dtype=tf.float32)
                    # W maps e_v (384 encoder / one column per category for one-hot) to the 50-d POI space
                    self.W_sem = tf.get_variable('W_sem', [self.sem_table.shape[1], self.args.embedding_size],
                                                 initializer=self.initializer)
                    # m_v: in training each POI's q_v is dropped with probability mu (redrawn every step),
                    # so the model also learns to rank from W e_v alone, as for a cold POI. Default 0 = test.
                    self.id_drop = tf.placeholder_with_default(0.0, shape=[], name='id_drop')
                    id_keep = tf.cast(tf.random_uniform([self.num_items + 1, 1]) >= self.id_drop, tf.float32)
                   # ** We rescale the meaning vector to to scaling factor s, previous runs made Wev dominate the loss and W parameters grew too large dominating also q_v
                    # We additonaly do l2 normalisation to Wev ev, sicnce simple 0.1 didnt prevent the weights from expoding in later epochs.
                    poi_vectors = id_keep * self.item_embeddings + self.args.scaling_sem * tf.nn.l2_normalize(tf.matmul(sem_table, self.W_sem), axis=1)                
                self.all_items_emb = tf.nn.embedding_lookup(poi_vectors, self.all_items)      # history [B, P, d]
                self.item_emb = tf.nn.embedding_lookup(poi_vectors, self.item_input)          # positive / candidate [B, d]
                self.item_emb_neg = tf.nn.embedding_lookup(poi_vectors, self.item_input_neg)  # negative [B, d]
                
                # ---- Geography parameters (paper 4.5), one scalar each:
                #      rho_u = dis_W[u], b_u = dis_b[u], rho_v = dis_W_item[v]  (all tied to ids)
                self.dis_W = tf.get_variable("W", [self.num_users + 1, 1], initializer=self.initializer)
                self.dis_b = tf.get_variable("b", [self.num_users + 1, 1], initializer=self.initializer)
                self.dis_W_item = tf.get_variable("W_item", [self.num_items + 1, 1], initializer=self.initializer)

                if self.args.constraint:
                    # Keep embeddings inside the unit ball (metric learning, as in LRML/CML).
                    # [CHANGED C20] fixes [NOTE N1]. upstream: axes=1. all_items_emb is [B, P, d], so axes=1 took
                    # the norm ACROSS the P = 200 history positions (per embedding dimension), which shrank every
                    # history vector to about 1/10 of its norm. axes=-1 (= 2) clips each POI vector, as for the
                    # positive and negative below ([B, d], where axes=1 is already the last axis).
                    # Must stay above the transpose below, after which the last axis is P.
                    self.all_items_emb = tf.clip_by_norm(self.all_items_emb, 1.0, axes=-1) # clipping changed to row level rather than column level
                    self.item_emb = tf.clip_by_norm(self.item_emb, 1.0, axes=1)
                    self.item_emb_neg = tf.clip_by_norm(self.item_emb_neg, 1.0, axes=1)

                # ---- User vector p_u = mean of the history embeddings (Eq. 2):
                # zero the padded positions with a length mask, sum over the history, divide by L.
                self.all_items_emb = tf.transpose(self.all_items_emb, perm=[0, 2, 1])
                self.cur_mask = tf.sequence_mask(self.L, self.args.max_p_num)
                self.cur_mask = tf.expand_dims(self.cur_mask, -1)
                self.cur_mask = tf.transpose(self.cur_mask, perm=[0, 2, 1])
                kept_indices = tf.cast(self.cur_mask, dtype=tf.float32)
                self.all_items_emb = self.all_items_emb * kept_indices
                self.user_emb_sum = tf.reduce_sum(self.all_items_emb, 2)
                self.LL = tf.expand_dims(self.L, -1)
                self.user_emb = self.user_emb_sum / self.LL

                # [CHANGED C21] fixes [NOTE N2]. upstream looked the positive up again here:
                #   self.item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
                # which overwrote the clipped positive with the raw one, so positives (and every test candidate)
                # were unclipped while training negatives were clipped. Line removed.



                #self.item_emb = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)

                # ---- TLDA link (paper 4.4, Eq. 12-14): predict the user's TLDA topic mixture (self.label)
                # from p_u with one softmax layer. This term only shapes the embeddings during training;
                # it is not part of the ranking score.
                self.user_topic_W = tf.Variable(
                    tf.random_normal([self.args.embedding_size, self.args.topic_num], stddev=stddev))
                self.user_topic_b = tf.Variable(tf.random_normal([self.args.topic_num], stddev=stddev))
                self.topic_out = tf.matmul(self.user_emb, self.user_topic_W) + self.user_topic_b
                self.predict_topic = tf.nn.softmax(self.topic_out)
                self.topic_cost = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.topic_out, labels=self.label))

                # ---- Key-value memory read (Eq. 3-5):
                # key e_uv = p_u * q_v; attention = softmax(e_uv K); relation r_uv = attention M
                self._key = tf.multiply(self.user_emb, self.item_emb)
                self.key_attention = tf.matmul(self._key, self.user_item_key)
                self.key_attention = tf.nn.softmax(self.key_attention)
                self.selected_memory = tf.matmul(self.key_attention, self.memories)
                # ---- Memory scores (Eq. 6). The negative reuses the POSITIVE pair's relation vector r_uv,
                # as the paper describes (4.2.3: it "empirically" worked better).
                final_layer = self._composition_layer(self.user_emb, self.item_emb,
                                                      selected_memory=self.selected_memory)
                final_layer_neg = self._composition_layer(self.user_emb, self.item_emb_neg,
                                                          selected_memory=self.selected_memory)
                # [CHANGED C5] upstream defined `self.predict_op = tf.squeeze(final_layer)` here (memory score
                # only); it now comes after the geography part, below.
                # Memory hinge loss (Eq. 7): max(0, s_neg - s_pos + margin lamb_m), summed over the batch
                self.mem_cost = tf.reduce_sum(tf.nn.relu((tf.squeeze(final_layer_neg - final_layer) + self.lamb_m)))

                # ---- Geography (Eq. 17-18). DIST / DIST_neg = distance from the user's centre c_u to the
                # positive / negative POI, computed in process_TEMN_data (see [CHANGED C7] there).
                self.dis_W_emb = tf.squeeze(tf.nn.embedding_lookup(self.dis_W, self.user_input))
                self.dis_b_emb = tf.squeeze(tf.nn.embedding_lookup(self.dis_b, self.user_input))
                self.dist_W_item_emb = tf.squeeze(tf.nn.embedding_lookup(self.dis_W_item, self.item_input))
                self.dist_W_item_emb_neg = tf.squeeze(tf.nn.embedding_lookup(self.dis_W_item, self.item_input_neg))
                # s_geo = rho_u * d + b_u + rho_v * d = (rho_u + rho_v) * d + b_u   (Eq. 17)
                self.Wis = (self.dis_W_emb * self.DIST + self.dis_b_emb + self.dist_W_item_emb * self.DIST)
                self.Wis_neg = (
                            self.dis_W_emb * self.DIST_neg + self.dis_b_emb + self.dist_W_item_emb_neg * self.DIST_neg)
                # Geography hinge loss (Eq. 18) with margin lamb_d
                self.dist_cost = tf.reduce_sum(tf.nn.relu((self.lamb_d - self.Wis + self.Wis_neg)))

                # [CHANGED C5] upstream: self.predict_op = tf.squeeze(final_layer) -- the memory score only,
                # so the geography module was trained (dist_cost) but never used to rank POIs.
                # Now the full TEMN score of the paper.
                # Final score (paper Eq. 19): s = s_mem + eta * s_geo
                # self.Wis is the geographic score (rho_u + rho_v) * d + b_u of the scored POI.
                # b_u is the same for every candidate of a user and cancels in the pairwise
                # loss: Wis - Wis_neg = (rho_u+rho_v)*d_v - (rho_u+rho_j)*d_j, so it never
                # changes a ranking.
                self.predict_op = tf.squeeze(final_layer, axis=1) + self.args.eta * self.Wis

                # ---- Joint objective (Eq. 20): ratio1 = topic weight (paper: varsigma), ratio2 = geography
                # weight (paper: epsilon), plus L2 on every trainable variable (the 'bias' filter matches no name).
                self.cost = self.mem_cost + self.topic_cost * self.ratio1 + self.dist_cost * self.ratio2
                if self.args.l2_reg > 0:
                    vars = tf.trainable_variables()
                    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * self.args.l2_reg
                    self.cost += lossL2
                if self.args.opt == 'SGD':
                    self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
                elif self.args.opt == 'Adam':
                    self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
                elif self.args.opt == 'Adadelta':
                    self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
                elif self.args.opt == 'Adagrad':
                    self.opt = tf.train.AdagradOptimizer(learning_rate=self.learn_rate, initial_accumulator_value=0.9)
                elif self.args.opt == 'RMS':
                    self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate, decay=0.9, epsilon=1e-6)
                elif self.args.opt == 'Moment':
                    self.opt = tf.train.MomentumOptimizer(self.args.learn_rate, 0.9)
                # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 1)
                gradients = self.opt.compute_gradients(self.cost)
                self.gradients = gradients

                def ClipIfNotNone(grad):
                    if grad is None:
                        return grad
                    # [CHANGED C3] new line (not in upstream).
                    # TF 1.12 densified sparse (IndexedSlices) gradients here; TF 2.x
                    # would clip the sparse slices instead, which changes the result.
                    grad = tf.convert_to_tensor(grad)
                    grad = tf.clip_by_value(grad, -10, 10, name=None)
                    return tf.clip_by_norm(grad, self.args.clip_norm)

                if self.args.clip_norm > 0:
                    clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gradients]
                else:
                    clipped_gradients = [(grad, var) for grad, var in gradients]

                # grads, _ = tf.clip_by_value(tf.gradients(self.cost, tvars),-10,10)
                self.optimizer = self.opt.apply_gradients(clipped_gradients)
                self.train_op = self.optimizer
