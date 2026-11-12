# encoding:utf-8
"""
xz331@cam.ac.uk
abnerzxzhao@tencent.com
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# tf.contrib was removed in TF 2.x. These reproduce the TF 1.x contrib
# initializers exactly: contrib's truncated normal used stddev = sqrt(1.3*factor/n),
# while variance_scaling_initializer uses sqrt(scale/n)/0.87962566103423978,
# so scale is adjusted to give the identical stddev.
_TRUNC_ADJ = 1.3 * 0.87962566103423978 ** 2


def _contrib_variance_scaling(factor, mode, uniform):
    if uniform:
        return tf.variance_scaling_initializer(scale=factor, mode=mode, distribution='uniform')
    return tf.variance_scaling_initializer(scale=factor * _TRUNC_ADJ, mode=mode,
                                           distribution='truncated_normal')


class basemodel(object):
    def __init__(self, args):
        self.args = args

    def _get_initializer(self):
        if self.args.init_method == 'tnormal':
            return tf.truncated_normal_initializer(stddev=self.args.stddev)
        elif self.args.init_method == 'uniform':
            return tf.random_uniform_initializer(-self.args.stddev, self.args.stddev)
        elif self.args.init_method == 'normal':
            return tf.random_normal_initializer(stddev=self.args.stddev)
        elif self.args.init_method == 'xavier_normal':
            return _contrib_variance_scaling(1.0, 'fan_avg', uniform=False)
        elif self.args.init_method == 'xavier_uniform':
            return _contrib_variance_scaling(1.0, 'fan_avg', uniform=True)
        elif self.args.init_method == 'he_normal':
            return _contrib_variance_scaling(2.0, 'fan_in', uniform=False)
        elif self.args.init_method == 'he_uniform':
            return _contrib_variance_scaling(2.0, 'fan_in', uniform=True)
        else:
            return tf.truncated_normal_initializer(stddev=self.args.stddev)

    def _set_opt(self):
        if self.args.opt == 'SGD':
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
        elif self.args.opt == 'Adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        elif self.args.opt == 'Adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
        elif self.args.opt == 'Adagrad':
            self.opt = tf.train.AdagradOptimizer(learning_rate=self.learn_rate,
                                                 initial_accumulator_value=0.9)
        elif self.args.opt == 'RMS':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate,
                                                 decay=0.9, epsilon=1e-6)
        elif self.args.opt == 'Moment':
            self.opt = tf.train.MomentumOptimizer(self.args.learn_rate, 0.9)
