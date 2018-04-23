import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np
import logging
import codecs
import random

from tensor2tensor.common_attention import multihead_attention, add_timing_signal_1d, attention_bias_ignore_padding, attention_bias_lower_triangle
from tensor2tensor.common_layers import layer_norm, conv_hidden_relu, smoothing_cross_entropy
from share_function import deal_generated_samples
from share_function import score
from share_function import remove_pad_tolist

INT_TYPE = np.int32
FLOAT_TYPE = np.float32


class Model(object):
    def __init__(self, config, graph=None, sess=None):
        if graph is None:
            self.graph=tf.Graph()
        else:
            self.graph = graph

        if sess is None:
            self.sess=tf.Session(graph=self.graph)
        else:
            self.sess=sess

        self.config = config
        self._logger = logging.getLogger('model')
        self._prepared = False
        self._summary = True

    def prepare(self, is_training):
        assert not self._prepared
        self.is_training = is_training
        # Select devices according to running is_training flag.
        devices = self.config.train.devices if is_training else self.config.test.devices
        self.devices = ['/gpu:'+i for i in devices.split(',')] or ['/cpu:0']
        # If we have multiple devices (typically GPUs), we set /cpu:0 as the sync device.
        self.sync_device = self.devices[0] if len(self.devices) == 1 else '/cpu:0'

        if is_training:
            with self.graph.as_default():
                with tf.device(self.sync_device):
                    # Preparing optimizer.
                    self.global_step = tf.get_variable(name='global_step', dtype=INT_TYPE, shape=[],
                                                       trainable=False, initializer=tf.zeros_initializer)
                    self.learning_rate = tf.convert_to_tensor(self.config.train.learning_rate)

                    self.gan_optimizer=tf.train.RMSPropOptimizer(self.config.train.gan_learning_rate)
                    if self.config.train.optimizer == 'adam':
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    elif self.config.train.optimizer == 'adam_decay':
                        self.learning_rate = learning_rate_decay(self.config, self.global_step)
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                beta1=0.9, beta2=0.98, epsilon=1e-9)
                    elif self.config.train.optimizer == 'sgd':
                        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    elif self.config.train.optimizer == 'mom':
                        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                    else:
                        logging.info("No optimizer is defined for the model")
                        raise ValueError
        self._initializer = init_ops.variance_scaling_initializer(scale=1, mode='fan_avg', distribution='uniform')
        # self._initializer = tf.uniform_unit_scaling_initializer()
        self._prepared = True

    def build_variational_train_model(self):
        self.prepare(is_training=True)
        with self.graph.as_default():
            #with tf.device(self.sync_device):
            self.src_a_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_a_pl')
            self.dst_a_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_a_pl')
            self.reward_a_pl = tf.placeholder(dtype=tf.float32, shape=[None, None], name='reward_a_pl')
            self.given_num_a_pl = tf.placeholder(dtype=INT_TYPE, shape=[], name='given_num_a_pl')

            self.src_b_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_b_pl')
            self.dst_b_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_b_pl')
            self.reward_b_pl = tf.placeholder(dtype=tf.float32, shape=[None, None], name='reward_b_pl')
            self.given_num_b_pl = tf.placeholder(dtype=INT_TYPE, shape=[], name='given_num_b_pl')

            Xs_a = split_tensor(self.src_a_pl, len(self.devices))
            Ys_a = split_tensor(self.dst_a_pl, len(self.devices))
            Rs_a = split_tensor(self.reward_a_pl, len(self.devices))
            Ms_a = [self.given_num_a_pl] * len(self.devices)

            Xs_b = split_tensor(self.src_b_pl, len(self.devices))
            Ys_b = split_tensor(self.dst_b_pl, len(self.devices))
            Rs_b = split_tensor(self.reward_b_pl, len(self.devices))
            Ms_b = [self.given_num_b_pl] * len(self.devices)

            #avg_loss = tf.get_variable('avg_loss', initializer=100.0, trainable=False)

            acc_aa_list, loss_aa_list, gv_aa_list=[], [], []
            acc_bb_list, loss_bb_list, gv_bb_list=[], [], []
            acc_ab_list, loss_ab_list, gv_ab_list=[], [], []
            acc_ba_list, loss_ba_list, gv_ba_list=[], [], []
            gan_loss_ab_list, gan_gv_ab_list=[], []
            gan_loss_ba_list, gan_gv_ba_list=[], []

            #cross_acc_list, cross_loss_list, cross_gv_list=[],[],[]

            #gan_loss_list, gan_gv_list=[], []
            generate_ab_list, generate_ba_list=[], []
            roll_generate_ab_list, roll_generate_ba_list=[], []

            cache = {}
            load = dict([(d, 0) for d in self.devices])

            for i,(X_a,Y_a,R_a,M_a,X_b,Y_b,R_b,M_b,device) in enumerate(zip(Xs_a,Ys_a,Rs_a,Ms_a,Xs_b,Ys_b,Rs_b,Ms_b,self.devices)):
                #with tf.device(lambda op: self.choose_device(op, device)):
                def daisy_chain_getter(getter, name, *args, **kwargs):
                    """Get a variable and cache in a daisy chain."""
                    device_var_key = (device, name)
                    if device_var_key in cache:
                        # if we have the variable on the correct device, return it.
                        return cache[device_var_key]
                    if name in cache:
                        # if we have it on a different device, copy it from the last device
                        v = tf.identity(cache[name])
                    else:
                        var = getter(name, *args, **kwargs)
                        v = tf.identity(var._ref())  # pylint: disable=protected-access
                    # update the cache
                    cache[name] = v
                    cache[device_var_key] = v
                    return v

                def balanced_device_setter(op):
                    """Balance variables to all devices."""
                    if op.type in {'Variable', 'VariableV2', 'VarHandleOp'}:
                        # return self._sync_device
                        min_load = min(load.values())
                        min_load_devices = [d for d in load if load[d] == min_load]
                        chosen_device = random.choice(min_load_devices)
                        load[chosen_device] += op.outputs[0].get_shape().num_elements()
                        return chosen_device
                    return device

                def identity_device_setter(op):
                    return device

                device_setter = balanced_device_setter

                with tf.variable_scope(tf.get_variable_scope(),
                                       initializer=self._initializer,
                                       custom_getter=daisy_chain_getter,
                                       reuse=None):
                    with tf.device(device_setter):
                
                        self._logger.info('Build model on %s' % device)

                        enc_a_out, enc_a_shared = self.variational_encoder(X_a,
                                                                     self.config.enc_layer_indep,
                                                                     self.config.enc_layer_share,
                                                                     self.config.src_vocab_size_a,
                                                                     scope='enc_a',
                                                                     reuse=i>0 or None,
                                                                     step='f')

                        dec_aa_out = self.variational_decoder(shift_right(Y_a),
                                                              enc_a_out,
                                                              self.config.dec_layer_indep,
                                                              self.config.dec_layer_share,
                                                              self.config.dst_vocab_size_a,
                                                              scope='dec_a',
                                                              reuse=i>0 or None,
                                                              step='f')

                        acc_aa, loss_aa = self.train_output(dec_aa_out,
                                                            Y_a,
                                                            enc_a_shared,
                                                            self.config.dst_vocab_size_a,
                                                            scope='out_a',
                                                            reuse=i>0 or None)

                        enc_b_out, enc_b_shared = self.variational_encoder(X_b,
                                                                    self.config.enc_layer_indep,
                                                                    self.config.enc_layer_share,
                                                                    self.config.src_vocab_size_b,
                                                                    scope='enc_b',
                                                                    reuse=i>0 or None,
                                                                    step='l')

                        dec_bb_out = self.variational_decoder(shift_right(Y_b),
                                                              enc_b_out,
                                                              self.config.dec_layer_indep,
                                                              self.config.dec_layer_share,
                                                              self.config.dst_vocab_size_b,
                                                              scope='dec_b',
                                                              reuse=i>0 or None,
                                                              step='l')

                        acc_bb, loss_bb = self.train_output(dec_bb_out,
                                                            Y_b,
                                                            enc_b_shared,
                                                            self.config.dst_vocab_size_b,
                                                            scope='out_b',
                                                            reuse=i>0 or None)

                        dec_ab_out = self.variational_decoder(shift_right(Y_b),
                                                              enc_a_out,
                                                              self.config.dec_layer_indep,
                                                              self.config.dec_layer_share,
                                                              self.config.dst_vocab_size_b,
                                                              scope='dec_b',
                                                              reuse=True)

                        dec_ba_out = self.variational_decoder(shift_right(Y_a),
                                                              enc_b_out,
                                                              self.config.dec_layer_indep,
                                                              self.config.dec_layer_share,
                                                              self.config.dst_vocab_size_a,
                                                              scope='dec_a',
                                                              reuse=True)

                        acc_ab, loss_ab = self.train_output(dec_ab_out,
                                                            Y_b,
                                                            enc_a_shared,
                                                            self.config.dst_vocab_size_b,
                                                            scope='out_b',
                                                            reuse=True)

                        acc_ba, loss_ba = self.train_output(dec_ba_out,
                                                            Y_a,
                                                            enc_b_shared,
                                                            self.config.dst_vocab_size_a,
                                                            scope='out_a',
                                                            reuse=True)

                        generate_ab = self.variational_generate(enc_a_out,
                                                                self.config.generate_maxlen,
                                                                self.config.dst_vocab_size_b,
                                                                scope_dec='dec_b',
                                                                scope_out='out_b')

                        generate_ba = self.variational_generate(enc_b_out,
                                                                self.config.generate_maxlen,
                                                                self.config.dst_vocab_size_a,
                                                                scope_dec='dec_a',
                                                                scope_out='out_a')

                        roll_generate_ab = self.variational_roll_generate(enc_a_out,
                                                                          Y_b,
                                                                          M_b,
                                                                          self.config.generate_maxlen,
                                                                          self.config.dst_vocab_size_b,
                                                                          scope_dec='dec_b',
                                                                          scope_out='out_b')

                        roll_generate_ba = self.variational_roll_generate(enc_b_out,
                                                                          Y_a,
                                                                          M_a,
                                                                          self.config.generate_maxlen,
                                                                          self.config.dst_vocab_size_a,
                                                                          scope_dec='dec_a',
                                                                          scope_out='out_a')

                        gan_loss_ab = self.gan_output(dec_ab_out, Y_b, R_b, self.config.dst_vocab_size_b, scope='out_b')
                        gan_loss_ba = self.gan_output(dec_ba_out, Y_a, R_a, self.config.dst_vocab_size_a, scope='out_a')

                        # the whole variables
                        whole_variables = tf.trainable_variables()
                        #print('whole_variable is \n')
                        #for vari in whole_variables:
                        #    print(vari)
                        #print(whole_variables)

                        # indepedent parameters for a
                        param_enc_a_embed = [param for param in whole_variables if 'enc_aembedding' in param.name]
                        param_enc_a_indep = [param for param in whole_variables if 'enc_aindep_block' in param.name]
                        param_dec_a_embed = [param for param in whole_variables if 'dec_aembedding' in param.name]
                        param_dec_a_indep = [param for param in whole_variables if 'dec_aindep_block' in param.name]
                        param_a_out = [param for param in whole_variables if 'out_a' in param.name]

                        # independent parameters for b
                        param_enc_b_embed = [param for param in whole_variables if 'enc_bembedding' in param.name]
                        param_enc_b_indep = [param for param in whole_variables if 'enc_bindep_block' in param.name]
                        param_dec_b_embed = [param for param in whole_variables if 'dec_bembedding' in param.name]
                        param_dec_b_indep = [param for param in whole_variables if 'dec_bindep_block' in param.name]
                        param_b_out = [param for param in whole_variables if 'out_b' in param.name]

                        # shared parameters for a and b
                        param_enc_shared = [param for param in whole_variables if 'enc_shared_block' in param.name]
                        param_dec_shared = [param for param in whole_variables if 'dec_shared_block' in param.name]

                        clac_param = (param_enc_a_embed + param_enc_a_indep + param_dec_a_embed + param_dec_a_indep + param_a_out + param_enc_b_embed +  
                                           param_enc_b_indep + param_dec_b_embed + param_dec_b_indep + param_b_out + param_enc_shared + param_dec_shared)
                        
                        if self.config.multi_channel_encoder:
                            param_enc_a_rnn_emb = [param for param in whole_variables if 'enc_arnn_emb' in param.name]
                            param_enc_b_rnn_emb = [param for param in whole_variables if 'enc_brnn_emb' in param.name]
                            clac_param = (param_enc_a_embed + param_enc_a_indep + param_dec_a_embed + param_dec_a_indep + param_a_out + param_enc_b_embed +  
                                    param_enc_b_indep + param_dec_b_embed + param_dec_b_indep + param_b_out + param_enc_shared + param_dec_shared) + param_enc_a_rnn_emb + param_enc_b_rnn_emb
                        #print('calc vari is \n')
                        #for vari in cla_param:
                        #    print(vari)
                        #print(cla_param)
                        assert(len(clac_param) == len(whole_variables))

                        # params for translating a to a
                        if self.config.lock_enc_embed:
                            param_aa = param_enc_a_indep + param_dec_a_embed + param_dec_a_indep + param_a_out + param_enc_shared + param_dec_shared
                            param_bb = param_enc_b_indep + param_dec_b_embed + param_dec_b_indep + param_b_out + param_enc_shared + param_dec_shared
                            param_ab = param_enc_a_indep + param_dec_b_embed + param_dec_b_indep + param_b_out + param_enc_shared + param_dec_shared
                            param_ba = param_enc_b_indep + param_dec_a_embed + param_dec_a_indep + param_a_out + param_enc_shared + param_dec_shared
                            if self.config.multi_channel_encoder:
                                param_aa += param_enc_a_rnn_emb
                                param_bb += param_enc_b_rnn_emb
                                param_ab += param_enc_a_rnn_emb
                                param_ba += param_enc_b_rnn_emb
                        else:
                            param_aa = param_enc_a_embed + param_enc_a_indep + param_dec_a_embed + param_dec_a_indep + param_a_out + param_enc_shared + param_dec_shared
                            param_bb = param_enc_b_embed + param_enc_b_indep + param_dec_b_embed + param_dec_b_indep + param_b_out + param_enc_shared + param_dec_shared
                            param_ab = param_enc_a_embed + param_enc_a_indep + param_dec_b_embed + param_dec_b_indep + param_b_out + param_enc_shared + param_dec_shared
                            param_ba = param_enc_b_embed + param_enc_b_indep + param_dec_a_embed + param_dec_a_indep + param_a_out + param_enc_shared + param_dec_shared

                            if self.config.multi_channel_encoder:
                                param_aa += param_enc_a_rnn_emb
                                param_bb += param_enc_b_rnn_emb
                                param_ab += param_enc_a_rnn_emb
                                param_ba += param_enc_b_rnn_emb

                        acc_aa_list.append(acc_aa)
                        loss_aa_list.append(loss_aa)
                        acc_bb_list.append(acc_bb)
                        loss_bb_list.append(loss_bb)

                        acc_ab_list.append(acc_ab)
                        loss_ab_list.append(loss_ab)
                        acc_ba_list.append(acc_ba)
                        loss_ba_list.append(loss_ba)

                        gan_loss_ab_list.append(gan_loss_ab)
                        gan_loss_ba_list.append(gan_loss_ba)

                        gv_aa_list.append(self.optimizer.compute_gradients(loss_aa, var_list=param_aa))
                        gv_bb_list.append(self.optimizer.compute_gradients(loss_bb, var_list=param_bb))

                        gv_ab_list.append(self.optimizer.compute_gradients(loss_ab, var_list=param_ab))
                        gv_ba_list.append(self.optimizer.compute_gradients(loss_ba, var_list=param_ba))

                        gan_gv_ab_list.append(self.gan_optimizer.compute_gradients(gan_loss_ab, var_list=param_ab))
                        gan_gv_ba_list.append(self.gan_optimizer.compute_gradients(gan_loss_ba, var_list=param_ba))

                        generate_ab_list.append(generate_ab)
                        generate_ba_list.append(generate_ba)
                        roll_generate_ab_list.append(roll_generate_ab)
                        roll_generate_ba_list.append(roll_generate_ba)

                        #gan_loss=0.5 * (gan_loss_ab+gan_loss_ba)
                        #gan_loss_list.append(gan_loss)
                        #gan_gv_list.append(self.gan_optimizer.compute_gradients(gan_loss))

            self.acc_aa = tf.reduce_mean(acc_aa_list)
            self.acc_bb = tf.reduce_mean(acc_bb_list)
            self.acc_ab = tf.reduce_mean(acc_ab_list)
            self.acc_ba = tf.reduce_mean(acc_ba_list)

            self.loss_aa = tf.reduce_mean(loss_aa_list)
            self.loss_bb = tf.reduce_mean(loss_bb_list)
            self.loss_ab = tf.reduce_mean(loss_ab_list)
            self.loss_ba = tf.reduce_mean(loss_ba_list)

            self.gan_loss_ab = tf.reduce_mean(gan_loss_ab_list)
            self.gan_loss_ba = tf.reduce_mean(gan_loss_ba_list)


            #self.acc = tf.reduce_mean(acc_list)
            #self.loss = tf.reduce_mean(loss_list)
            #self.mov_loss = avg_loss.assign(avg_loss * 0.9 + self.loss * 0.1)

            #self.cross_acc = tf.reduce_mean(cross_acc_list)
            #self.cross_loss = tf.reduce_mean(cross_loss_list)

            #self.gan_loss = tf.reduce_mean(gan_loss_list)

            self.generate_ab = tf.concat(generate_ab_list, axis=0)
            self.generate_ba = tf.concat(generate_ba_list, axis=0)
            self.roll_generate_ab = tf.concat(roll_generate_ab_list, axis=0)
            self.roll_generate_ba = tf.concat(roll_generate_ba_list, axis=0)

            
            #grads_and_vars = average_gradients(gv_list)
            #cross_grads_and_vars = average_gradients(cross_gv_list)
            #gan_grads_and_vars = average_gradients(gan_gv_list)

            grads_and_vars_aa = average_gradients(gv_aa_list)
            grads_and_vars_bb = average_gradients(gv_bb_list)
            grads_and_vars_ab = average_gradients(gv_ab_list)
            grads_and_vars_ba = average_gradients(gv_ba_list)

            
            gan_grads_and_vars_ab = average_gradients(gan_gv_ab_list)
            gan_grads_and_vars_ba = average_gradients(gan_gv_ba_list)

            #if self._summary:
            #    for g, v in grads_and_vars:
            #        tf.summary.histogram('variables/' + v.name.split(':')[0], v)
            #        tf.summary.histogram('gradients/' + v.name.split(':')[0], g)

            # simple loss
            grads_aa, self.grads_norm_aa = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars_aa],
                                                            clip_norm=self.config.train.grads_clip)
            grads_and_vars_aa = zip(grads_aa, [gv[1] for gv in grads_and_vars_aa])
            self.train_op_aa = self.optimizer.apply_gradients(grads_and_vars_aa, global_step=self.global_step)

            grads_bb, self.grads_norm_bb = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars_bb],
                                                            clip_norm=self.config.train.grads_clip)
            grads_and_vars_bb = zip(grads_bb, [gv[1] for gv in grads_and_vars_bb])
            self.train_op_bb = self.optimizer.apply_gradients(grads_and_vars_bb, global_step=self.global_step)

            # cross loss
            grads_ab, self.grads_norm_ab = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars_ab],
                                                            clip_norm=self.config.train.grads_clip)
            grads_and_vars_ab = zip(grads_ab, [gv[1] for gv in grads_and_vars_ab])
            self.train_op_ab = self.optimizer.apply_gradients(grads_and_vars_ab, global_step=self.global_step)

            grads_ba, self.grads_norm_ba = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars_ba],
                                                            clip_norm=self.config.train.grads_clip)
            grads_and_vars_ba = zip(grads_ba, [gv[1] for gv in grads_and_vars_ba])
            self.train_op_ba = self.optimizer.apply_gradients(grads_and_vars_ba, global_step=self.global_step)

            gan_grads_ab, self.gan_grads_norm_ab = tf.clip_by_global_norm([gv[0] for gv in gan_grads_and_vars_ab],
                                                            clip_norm=self.config.train.grads_clip)
            gan_grads_and_vars_ab = zip(gan_grads_ab, [gv[1] for gv in gan_grads_and_vars_ab])
            self.gan_train_op_ab = self.optimizer.apply_gradients(gan_grads_and_vars_ab, global_step=self.global_step)

            gan_grads_ba, self.gan_grads_norm_ba = tf.clip_by_global_norm([gv[0] for gv in gan_grads_and_vars_ba],
                                                            clip_norm=self.config.train.grads_clip)
            gan_grads_and_vars_ba = zip(gan_grads_ba, [gv[1] for gv in gan_grads_and_vars_ba])
            self.gan_train_op_ba = self.optimizer.apply_gradients(gan_grads_and_vars_ba, global_step=self.global_step)


            #cross_grads, cross_grads_norm = tf.clip_by_global_norm([gv[0] for gv in cross_grads_and_vars],
            #                                                    clip_norm=self.config.train.grads_clip)
            #cross_grads_and_vars = zip(cross_grads, [gv[1] for gv in cross_grads_and_vars])
            #self.cross_train_op = self.optimizer.apply_gradients(cross_grads_and_vars, global_step=self.global_step)

            ## gan loss
            #gan_grads, gan_grads_norm = tf.clip_by_global_norm([gv[0] for gv in gan_grads_and_vars],
            #                                                   clip_norm=self.config.train.grads_clip)
            #gan_grads_and_vars = zip(gan_grads, [gv[1] for gv in gan_grads_and_vars])
            #self.gan_train_op = self.gan_optimizer.apply_gradients(gan_grads_and_vars,
            #                                                       global_step=self.global_step)

            # Summaries
            tf.summary.scalar('acc_aa', self.acc_aa)
            tf.summary.scalar('acc_bb', self.acc_bb)
            tf.summary.scalar('acc_ab', self.acc_ab)
            tf.summary.scalar('acc_ba', self.acc_ba)

            tf.summary.scalar('loss_aa', self.loss_aa)
            tf.summary.scalar('loss_bb', self.loss_bb)
            tf.summary.scalar('loss_ab', self.loss_ab)
            tf.summary.scalar('loss_ba', self.loss_ba)

            #tf.summary.scalar('loss', self.loss)
            #tf.summary.scalar('loss', self.mov_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)

            tf.summary.scalar('grads_norm_aa', self.grads_norm_aa)
            tf.summary.scalar('grads_norm_bb', self.grads_norm_bb)
            tf.summary.scalar('grads_norm_ab', self.grads_norm_ab)
            tf.summary.scalar('grads_norm_ba', self.grads_norm_ba)

            self.summary_op = tf.summary.merge_all()

    def build_variational_test_model(self, mode='ab'):
        self.prepare(is_training=False)

        self.mode = mode

        if self.mode == 'ab':
            logging.info('translating source to target')
            scope_encoder = 'enc_a'
            scope_decoder = 'dec_b'
            scope_out = 'out_b'
            vocab_size_enc = self.config.src_vocab_size_a
            vocab_size = self.config.dst_vocab_size_b
        elif self.mode == 'aa':
            logging.info('translating source to source')
            scope_encoder = 'enc_a'
            scope_decoder = 'dec_a'
            scope_out = 'out_a'
            vocab_size_enc = self.config.src_vocab_size_a
            vocab_size = self.config.dst_vocab_size_a
        elif self.mode == 'ba':
            logging.info('translating target to source')
            scope_encoder = 'enc_b'
            scope_decoder = 'dec_a'
            scope_out = 'out_a'
            vocab_size_enc = self.config.src_vocab_size_b
            vocab_size = self.config.dst_vocab_size_a
        elif self.mode == 'bb':
            logging.info('translating target to target')
            scope_encoder = 'enc_b'
            scope_decoder = 'dec_b'
            scope_out = 'out_b'
            vocab_size_enc = self.config.src_vocab_size_b
            vocab_size = self.config.dst_vocab_size_b
        else:
            raise Exception('mode error!')

        with self.graph.as_default():
            with tf.device(self.sync_device):
                self.src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_pl')
                self.dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_pl')
                self.decoder_input=shift_right(self.dst_pl)

                Xs = split_tensor(self.src_pl, len(self.devices))
                Ys = split_tensor(self.dst_pl, len(self.devices))
                dec_inputs = split_tensor(self.decoder_input, len(self.devices))

                encoder_output_list=[]

                for i, (X, device) in enumerate(zip(Xs, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        enc_out, enc_shared = self.variational_encoder(X, self.config.enc_layer_indep,
                                self.config.enc_layer_share, vocab_size_enc, scope=scope_encoder, reuse=i>0 or None, step='f')
                        encoder_output_list.append(enc_out)

                    self.encoder_output=tf.concat(encoder_output_list, axis=0)

                enc_outputs = split_tensor(self.encoder_output, len(self.devices))
                preds_list, k_preds_list, k_scores_list= [],[],[]
                self.loss_sum=0.0

                for i, (enc_output, dec_input, Y, device) in enumerate(zip(enc_outputs, dec_inputs, Ys, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        self._logger.info('Build model on %s.'  % device)
                        dec_b_out = self.variational_decoder(dec_input, enc_output, self.config.dec_layer_indep,
                                               self.config.dec_layer_share, vocab_size, scope=scope_decoder, reuse=i>0 or None, step='f')

                        preds, k_preds, k_scores = self.test_output(dec_b_out, vocab_size, scope=scope_out, reuse=i>0 or None)
                        preds_list.append(preds)
                        k_preds_list.append(k_preds)
                        k_scores_list.append(k_scores)

                        loss = self.test_loss(dec_b_out, Y, vocab_size, scope=scope_out, reuse=True)
                        self.loss_sum += loss

                self.preds = tf.concat(preds_list, axis=0)
                self.k_preds = tf.concat(k_preds_list, axis=0)
                self.k_scores = tf.concat(k_scores_list, axis=0)

    def variational_encoder(self, input_x, layers_indep, layers_share, vocab_size, scope="encoder", reuse=None, step='f'):
        encoder_padding = tf.equal(input_x, 0)
        encoder_output=bottom(input_x,
                              vocab_size=vocab_size,
                              dense_size=self.config.hidden_units,
                              scope=scope+'embedding',
                              shared_embedding=self.config.train.shared_embedding,
                              reuse=reuse,
                              multiplier=self.config.hidden_units**0.5 if self.config.scale_embedding else 1.0)

        encoder_output_vari=bottom(input_x,
                              vocab_size=vocab_size,
                              dense_size=self.config.hidden_units,
                              scope=scope+'variEmbed',
                              shared_embedding=self.config.train.shared_embedding,
                              reuse=reuse,
                              multiplier=self.config.hidden_units**0.5 if self.config.scale_embedding else 1.0)

        encoder_output = self.config.vari_emb_scale * encoder_output_vari + (1 - self.config.vari_emb_scale) * encoder_output

       
        embedding = encoder_output

        with tf.variable_scope(scope, initializer=self._initializer, reuse=reuse):
            encoder_output=add_timing_signal_1d(encoder_output)
            encoder_output = tf.layers.dropout(encoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)

            encoder_output=self.build_enc_block(encoder_output, encoder_padding, layers_indep,
                                                scope+'indep_block', reuse=reuse)

        encoder_output=self.build_enc_block(encoder_output, encoder_padding, layers_share,
                                            "enc_shared_block", reuse=reuse if step=='f' else True)

        encoder_shared=tf.reduce_mean(encoder_output,axis=1)
        encoder_shared +=tf.random_normal(shape=tf.shape(encoder_shared),
                                           mean=0.0,
                                           stddev=1.0,
                                           dtype=tf.float32,
                                           name=scope+'normal_random')

        if self.config.multi_channel_encoder:
            encoder_output = self.build_rnn_emb(embedding, encoder_output, reuse_var=reuse, scope=scope+'rnn_emb')

        return encoder_output,encoder_shared

    def variational_decoder(self, input_x, enc_output, layers_indep, layers_share, vocab_size, scope='decoder', reuse=None, step='f'):

        encoder_padding=tf.equal(tf.reduce_sum(tf.abs(enc_output), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        decoder_output=target(input_x,
                              vocab_size=vocab_size,
                              dense_size=self.config.hidden_units,
                              scope=scope+'embedding',
                              shared_embedding=self.config.train.shared_embedding,
                              reuse=reuse,
                              multiplier=self.config.hidden_units**0.5 if self.config.scale_embedding else 1.0)

        with tf.variable_scope(scope, initializer=self._initializer, reuse=reuse):
            decoder_output += add_timing_signal_1d(decoder_output)

            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)

            self_attention_bias=attention_bias_lower_triangle(tf.shape(decoder_output)[1])

        decoder_output = self.build_dec_block(decoder_output, enc_output, encoder_attention_bias,
                        self_attention_bias, layers_share, "dec_shared_block", reuse=reuse if step=='f' else True)

        decoder_output = self.build_dec_block(decoder_output, enc_output, encoder_attention_bias,
                        self_attention_bias, layers_indep, scope+"indep_block", reuse=reuse)

        return decoder_output

    def variational_generate(self, encoder_output, max_len, vocab_size, scope_dec, scope_out):

        batch_size=tf.shape(encoder_output)[0]

        def recurrency(i, cur_y, encoder_output):
            decoder_output = self.variational_decoder(shift_right(cur_y), encoder_output, self.config.dec_layer_indep,
                                    self.config.dec_layer_share, vocab_size, scope=scope_dec, reuse=True)

            next_logits = top(body_output=decoder_output,
                              vocab_size=vocab_size,
                              dense_size=self.config.hidden_units,
                              scope=scope_out,
                              shared_embedding=self.config.train.shared_embedding,
                              reuse=True)

            next_logits = next_logits[:, i, :]
            next_logits = tf.reshape(next_logits, [-1, vocab_size])
            next_probs = tf.nn.softmax(next_logits)
            next_sample = tf.argmax(next_probs, 1)
            next_sample = tf.expand_dims(next_sample, -1)
            next_sample = tf.to_int32(next_sample)
            next_y = tf.concat([cur_y[:, :i], next_sample], axis=1)
            next_y = tf.pad(next_y, [[0, 0], [0, max_len - 1 - i]])
            next_y.set_shape([None, max_len])
            return i + 1, next_y, encoder_output

        initial_y = tf.zeros((batch_size, max_len), dtype=INT_TYPE)  ##begin with <s>
        initial_i = tf.constant(0, dtype=tf.int32)
        _, sample_result, _ = tf.while_loop(
            cond=lambda a, _1, _2: a < max_len,
            body=recurrency,
            loop_vars=(initial_i, initial_y, encoder_output),
            shape_invariants=(initial_i.get_shape(), initial_y.get_shape(), encoder_output.get_shape())
        )

        return sample_result

    def variational_roll_generate(self, encoder_output, Y, given_num_pl, max_len, vocab_size, scope_dec, scope_out):
        batch_size = tf.shape(encoder_output)[0]

        def recurrency(given_num, given_y, encoder_output):
            decoder_output = self.variational_decoder(shift_right(given_y),
                                                      encoder_output,
                                                      self.config.dec_layer_indep,
                                                      self.config.dec_layer_share,
                                                      vocab_size,
                                                      scope=scope_dec,
                                                      reuse=True)

            next_logits = top(body_output=decoder_output,
                              vocab_size = vocab_size,
                              dense_size = self.config.hidden_units,
                              scope=scope_out,
                              shared_embedding = self.config.train.shared_embedding,
                              reuse=True)

            next_logits = next_logits[:, given_num, :]
            next_probs = tf.nn.softmax(next_logits)
            log_probs = tf.log(next_probs)
            next_sample = tf.multinomial(log_probs, 1)
            next_sample_flat = tf.cast(next_sample, tf.int32)
            next_y = tf.concat([given_y[:, :given_num], next_sample_flat], axis=1)
            next_y = tf.pad(next_y, [[0, 0], [0, max_len - given_num -1]])
            next_y.set_shape([None, max_len])
            return given_num +1, next_y, encoder_output

        given_y = Y[:,:given_num_pl]

        init_given_y = tf.pad(given_y, [[0, 0], [0, (max_len-given_num_pl)]])
        _, roll_sample, _ = tf.while_loop(
            cond = lambda a, _1, _2: a < max_len,
            body=recurrency,
            loop_vars=(given_num_pl, init_given_y, encoder_output),
            shape_invariants=(given_num_pl.get_shape(), init_given_y.get_shape(), encoder_output.get_shape())
        )
        return roll_sample

    def build_enc_block(self, input_x, input_padding, layers, scope, reuse=False):
        encoder_padding = input_padding
        encoder_output = input_x

        with tf.variable_scope(scope, initializer=self._initializer, reuse=reuse):
            # encoder_output = tf.layers.dropout(encoder_output,
            #                                    rate=self.config.residual_dropout_rate,
            #                                    training=self.is_training)
            for i in range(layers):
                with tf.variable_scope("block_{}".format(i)):
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query_antecedent=encoder_output,
                                                  memory_antecedent=None,
                                                  bias=attention_bias_ignore_padding(encoder_padding),
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='encoder_self_attention',
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
        return encoder_output


    def build_dec_block(self, input_x, enc_output, enc_attention_bias, input_attenion_bias, layers, scope, reuse=False):
        self_attention_bias = input_attenion_bias
        decoder_output = input_x
        encoder_attention_bias=enc_attention_bias

        with tf.variable_scope(scope, initializer=self._initializer, reuse=reuse):
            # decoder_output = tf.layers.dropout(decoder_output,
            #                                    rate=self.config.residual_dropout_rate,
            #                                    training=self.is_training)

            for i in range(layers):
                with tf.variable_scope("block_{}".format(i)):
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query_antecedent=decoder_output,
                                                  memory_antecedent=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='decoder_self_attention',
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query_antecedent=decoder_output,
                                                  memory_antecedent=enc_output,
                                                  bias=encoder_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='decoder_vanilla_attention',
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)


                    decoder_output = residual(decoder_output,
                                              conv_hidden_relu(
                                                  inputs=decoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)
        return decoder_output

    def build_rnn_emb(self, embedding, enc_out, reuse_var=None, scope='rnn_emb'):

        with tf.variable_scope(scope, initializer=self._initializer, reuse=reuse_var):

            def linear_3d(inputs, use_bias, scope='linear', reuse_var=None):
                if reuse_var == True:
                    tf.get_variable_scope().reuse_variables()
                if not scope:
                    scope = tf.get_variable_scope()
                input_shape = inputs.get_shape()
                output_shape = tf.shape(inputs)
                dtype = inputs.dtype

                if len(input_shape) == 3:
                    input_size = output_size = input_shape[2].value
                    inputs = tf.reshape(inputs, [-1, output_size])
                else:
                    input_size = output_size = input_shape[1].value

                with tf.variable_scope(scope):
                    weights = tf.get_variable('weights', [input_size, output_size], dtype=dtype)
                    res = tf.matmul(inputs, weights)
                    if use_bias:
                        biases = tf.get_variable('bias', [output_size], dtype=dtype)
                        res += bias


                return tf.reshape(res, shape=output_shape)

            embedding_for_gate = linear_3d(embedding, use_bias=False, scope='embed_linear', reuse_var=reuse_var)
            enc_out_for_gate = linear_3d(enc_out, use_bias=False, scope='enc_linear', reuse_var=reuse_var)

            gate = tf.nn.sigmoid(embedding_for_gate + enc_out_for_gate)
            rnn_emb = gate * embedding + (1 - gate) * enc_out

        return rnn_emb

    def generate_step(self, sentence_x, sentence_y):
        generate_ab, generate_ba = self.sess.run(
           [self.generate_ab, self.generate_ba],
           feed_dict={self.src_a_pl:sentence_x, self.src_b_pl:sentence_y})
        return generate_ab, generate_ba

    def generate_step_and_update(self, sentence_x, sentence_y, reward, direction='ab'):
        if direction == 'ab':
            x_pl = self.src_a_pl
            y_pl = self.dst_b_pl
            reward_pl = self.reward_b_pl
            loss =  self.gan_loss_ab
            train_op = self.gan_train_op_ab
        elif direction == 'ba':
            x_pl = self.src_b_pl
            y_pl = self.dst_a_pl
            reward_pl = self.reward_a_pl
            loss = self.gan_loss_ba
            train_op = self.gan_train_op_ba
        else:
            raise Exception('direction error!')

        feed={x_pl:sentence_x, y_pl:sentence_y, reward_pl:reward}
        loss, _ = self.sess.run([loss, train_op], feed_dict=feed)
        return loss

    def generate_and_save(self, data_util, infile, generate_batch, outfile, direction='ab'):
        if direction == 'ab':
            x_pl = self.src_a_pl
            y_out = self.generate_ab
            ind_to_word='dst'
            dst2idx = data_util.dst2idx
        elif direction == 'ba':
            x_pl = self.src_b_pl
            y_out = self.generate_ba
            ind_to_word='src'
            dst2idx = data_util.src2idx
        else:
            raise Exception('direction error')
          
        
        outfile = codecs.open(outfile, 'w', 'utf-8')
        for batch in data_util.get_test_batches(infile, generate_batch):
            feed={x_pl:batch}
            out_generate=self.sess.run(y_out, feed_dict=feed)
            out_generate_dealed, _ = deal_generated_samples(out_generate, dst2idx)

            y_strs=data_util.indices_to_words_del_pad(out_generate_dealed, ind_to_word)

            for y_str in y_strs:
                outfile.write(y_str+'\n')
        outfile.close()

    def get_reward(self, x, x_to_maxlen, y_sample, y_sample_mask, rollnum, disc, max_len=50, bias_num=None, data_util=None, direction='ab'):
        
        if direction == 'ab':
            x_pl = self.src_a_pl
            y_pl = self.dst_b_pl
            given_num_pl = self.given_num_b_pl
            roll_output = self.roll_generate_ab
            target_index = 1
        elif direction == 'ba':
            x_pl = self.src_b_pl
            y_pl = self.dst_a_pl
            given_num_pl = self.given_num_a_pl
            roll_output = self.roll_generate_ba
            target_index = 1
        else:
            raise Exception('direction error!')

        rewards=[]
        x_to_maxlen=np.transpose(x_to_maxlen)

        for i in range(rollnum):
            for give_num in np.arange(1, max_len, dtype='int32'):
                feed={x_pl:x, y_pl:y_sample, given_num_pl:give_num}
                output = self.sess.run(roll_output, feed_dict=feed)
                
                #print(output.shape)
                #print(y_sample_mask.shape)

                #print("the sample is ", data_util.indices_to_words(y_sample))
                #print("the roll_sample result is ", data_util.indices_to_words(output))

                output=output * y_sample_mask
                #print("the roll aftter sample_mask is", data_util.indices_to_words(output))
                output=np.transpose(output)

                feed={disc.dis_input_x:output, disc.dis_dropout_keep_prob:1.0}
                ypred_for_auc=self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)

                ypred=np.array([item[target_index] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[give_num -1]+=ypred

            y_sample_transpose = np.transpose(y_sample)

            feed = {disc.dis_input_x:y_sample_transpose, disc.dis_dropout_keep_prob:1.0}
            ypred_for_auc=self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)
            ypred= np.array([item[target_index] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
            else:
                rewards[max_len -1]+=ypred

        rewards = np.transpose(np.array(rewards)) ## now rewards: batch_size * max_len

        if  bias_num is None:
            rewards = rewards * y_sample_mask  
            rewards = rewards / (1. * rollnum)
        else:
            bias = np.zeros_like(rewards)
            bias +=bias_num * rollnum
            rewards_minus_bias = rewards-bias

            rewards=rewards_minus_bias * y_sample_mask
            rewards = rewards / (1. * rollnum)
        return rewards

    def init_and_restore(self, modelFile=None):
        params = tf.trainable_variables()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(params)

        self.sess.run(init_op)
        self.saver = saver
        if modelFile is None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.config.train.logdir))
        else:
            self.saver.restore(self.sess, modelFile)

    def choose_device(self, op, device):
        """Choose a device according the op's type."""
        if op.type.startswith('Variable'):
            return self.sync_device
        return device

    def test_output(self, decoder_output, vocab_size, scope, reuse):
        last_logits = top(body_output=decoder_output[:, -1],
                     vocab_size = vocab_size,
                     dense_size = self.config.hidden_units,
                     scope=scope,
                     shared_embedding = self.config.train.shared_embedding,
                     reuse=reuse)
        """During test, we only need the last prediction."""
        with tf.variable_scope(scope,initializer=self._initializer,  reuse=reuse):
            #last_logits = tf.layers.dense(decoder_output[:,-1], self.config.dst_vocab_size)
            last_preds = tf.to_int32(tf.arg_max(last_logits, dimension=-1))
            z = tf.nn.log_softmax(last_logits)
            last_k_scores, last_k_preds = tf.nn.top_k(z, k=self.config.test.beam_size, sorted=False)
            last_k_preds = tf.to_int32(last_k_preds)
        return last_preds, last_k_preds, last_k_scores

    def test_loss(self, decoder_output, Y, vocab_size, scope, reuse):
        logits = top(body_output=decoder_output,
                     vocab_size = vocab_size,
                     dense_size = self.config.hidden_units,
                     scope=scope,
                     shared_embedding = self.config.train.shared_embedding,
                     reuse=reuse)

        with tf.variable_scope("output", initializer=self._initializer, reuse=reuse):
            #logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            mask = tf.to_float(tf.not_equal(Y, 0))
            labels = tf.one_hot(Y, depth=vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_sum = tf.reduce_sum(loss * mask)
        return loss_sum

    def gan_output(self, decoder_output, Y, reward, vocab_size, scope, reuse=True):
        logits = top(body_output=decoder_output,
                     vocab_size = vocab_size,
                     dense_size = self.config.hidden_units,
                     scope=scope,
                     shared_embedding = self.config.train.shared_embedding,
                     reuse=reuse)

        with tf.variable_scope(scope, initializer=self._initializer, reuse=reuse):
            l_shape=tf.shape(logits)
            probs = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
            probs = tf.reshape(probs, [l_shape[0], l_shape[1], l_shape[2]])
            sample = tf.to_float(l_shape[0])

            g_loss = -tf.reduce_sum(
                tf.reduce_sum(tf.one_hot(tf.reshape(Y, [-1]), vocab_size, 1.0, 0.0) *
                              tf.reshape(probs, [-1, vocab_size]), 1) *
                              tf.reshape(reward, [-1]), 0) / sample
        return g_loss

    def train_output(self, decoder_output, Y, enc_output, vocab_size, scope, reuse):
        """Calculate loss and accuracy."""
        logits = top(body_output=decoder_output,
                     vocab_size = vocab_size,
                     dense_size = self.config.hidden_units,
                     scope=scope,
                     shared_embedding = self.config.train.shared_embedding,
                     reuse=reuse)

        with tf.variable_scope(scope, initializer=self._initializer, reuse=reuse):
            #logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
            mask = tf.to_float(tf.not_equal(Y, 0))
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)

            # Smoothed loss
            loss = smoothing_cross_entropy(logits=logits, labels=Y, vocab_size=vocab_size,
                                           confidence=1-self.config.train.label_smoothing)
            mean_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask))

            kl_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(enc_output, 2), axis=-1))

            mean_loss = mean_loss + self.config.kl_weight * kl_loss

        return acc, mean_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def residual(inputs, outputs, dropout_rate, is_training):
    """Residual connection.

    Args:
        inputs: A Tensor.
        outputs: A Tensor.
        dropout_rate: A float.
        is_training: A bool.

    Returns:
        A Tensor.
    """
    output = inputs + tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
    output = layer_norm(output)
    return output


def split_tensor(input, n):
    """
    Split the tensor input to n tensors.
    Args:
        inputs: A tensor with size [b, ...].
        n: A integer.

    Returns: A tensor list, each tensor has size [b/n, ...].
    """
    batch_size = tf.shape(input)[0]
    ls = tf.cast(tf.lin_space(0.0, tf.cast(batch_size, FLOAT_TYPE), n + 1), INT_TYPE)
    return [input[ls[i]:ls[i+1]] for i in range(n)]


def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.train.learning_rate_warmup_steps)
    global_step = tf.to_float(global_step)
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def shift_right(input, pad=2):
    """Shift input tensor right to create decoder input. '2' denotes <S>"""
    return tf.concat((tf.ones_like(input[:, :1]) * pad, input[:, :-1]), 1)

def get_weight(vocab_size, dense_size, name=None):
     weights = tf.get_variable("kernel", [vocab_size, dense_size], initializer=tf.random_normal_initializer(0.0, 512**-0.5))
     return weights

def bottom(x, vocab_size, dense_size, scope='embedding', shared_embedding=True, reuse=None, multiplier=1.0):
    with tf.variable_scope(scope, reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=None):
               embedding_var = get_weight(vocab_size, dense_size)
               emb_x = tf.gather(embedding_var, x)
               if multiplier != 1.0:
                   emb_x *= multiplier
        else:
            with tf.variable_scope("src_embedding", reuse=None):
                embedding_var = get_weight(vocab_size, dense_size)
                emb_x = tf.gather(embedding_var, x)
                if multiplier !=1.0:
                    emb_x *= multiplier
    return emb_x

def target(x, vocab_size, dense_size, scope='embedding', shared_embedding=True, reuse=None, multiplier=1.0):
    with tf.variable_scope(scope, reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=True):
               embedding_var = get_weight(vocab_size, dense_size)
               emb_x = tf.gather(embedding_var, x)
               if multiplier != 1.0:
                   emb_x *= multiplier
        else:
            with tf.variable_scope("dst_embedding", reuse=None):
                embedding_var = get_weight(vocab_size, dense_size)
                emb_x = tf.gather(embedding_var, x)
                if multiplier !=1.0:
                    emb_x *= multiplier
    return emb_x

def top(body_output, vocab_size, dense_size, scope='embedding', shared_embedding=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=True):
                shape=tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, dense_size])
                embedding_var = get_weight(vocab_size, dense_size)
                logits = tf.matmul(body_output, embedding_var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [vocab_size]], 0))
        else:
            with tf.variable_scope("softmax", reuse=None):
                embedding_var = get_weight(vocab_size, dense_size)
                shape=tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, dense_size]) 
                logits = tf.matmul(body_output, embedding_var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [vocab_size]], 0))
    return logits

def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors."""
    with tf.variable_scope(
        name, default_name="embedding", values=[x], reuse=reuse):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        emb_x = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            emb_x *= multiplier
        return emb_x
