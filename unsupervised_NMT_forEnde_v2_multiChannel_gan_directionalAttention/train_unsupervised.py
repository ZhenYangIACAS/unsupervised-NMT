from __future__ import print_function
import yaml
import time
import os
import logging
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model
from share_function import deal_generated_samples
import codecs


def train(config):
    logger = logging.getLogger('')

    """Train a model with a config file."""
    du = DataUtil(config=config)
    du.load_vocab(src_vocab=config.src_vocab,
                  dst_vocab=config.dst_vocab,
                  src_vocab_size=config.src_vocab_size_a,
                  dst_vocab_size=config.src_vocab_size_b)

    model = Model(config=config)
    model.build_variational_train_model()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    with model.graph.as_default():
        saver = tf.train.Saver(var_list=tf.global_variables())
        summary_writer = tf.summary.FileWriter(config.train.logdir, graph=model.graph)
        # saver_partial = tf.train.Saver(var_list=[v for v in tf.trainable_variables() if 'Adam' not in v.name])

        with tf.Session(config=sess_config) as sess:
            # Initialize all variables.
            sess.run(tf.global_variables_initializer())
            reload_pretrain_embedding=False
            try:
                # saver_partial.restore(sess, tf.train.latest_checkpoint(config.train.logdir))
                # print('Restore partial model from %s.' % config.train.logdir)
                saver.restore(sess, tf.train.latest_checkpoint(config.train.logdir))
            except:
                logger.info('Failed to reload model.')
                reload_pretrain_embedding=True

            if reload_pretrain_embedding:
                logger.info('reload the pretrained embeddings for the encoders')
                src_pretrained_embedding={}
                dst_pretrained_embedding={}
                try:

                    for l in codecs.open(config.train.src_pretrain_wordemb_path, 'r', 'utf-8'):
                        word_emb=l.strip().split()
                        # print(word_emb)
                        if len(word_emb)== config.hidden_units + 1:
                            word, emb = word_emb[0], np.array(map(float, word_emb[1:]))
                            src_pretrained_embedding[word]=emb

                    for l in codecs.open(config.train.dst_pretrain_wordemb_path, 'r', 'utf-8'):
                        word_emb=l.strip().split()
                        if len(word_emb)==config.hidden_units + 1:
                            word, emb = word_emb[0], np.array(map(float, word_emb[1:]))
                            dst_pretrained_embedding[word]=emb

                    logger.info('reload the word embedding done')

                    tf.get_variable_scope().reuse_variables()
                    src_embed_a=tf.get_variable('enc_aembedding/src_embedding/kernel')
                    src_embed_b=tf.get_variable('enc_bembedding/src_embedding/kernel')

                    dst_embed_a=tf.get_variable('dec_aembedding/dst_embedding/kernel')
                    dst_embed_b=tf.get_variable('dec_bembedding/dst_embedding/kernel')

                    count_a=0
                    src_value_a=sess.run(src_embed_a)
                    dst_value_a=sess.run(dst_embed_a)
                    # print(src_value_a)
                    for word in src_pretrained_embedding:
                        if word in du.src2idx:
                            id = du.src2idx[word]
                            # print(id)
                            src_value_a[id] = src_pretrained_embedding[word]
                            dst_value_a[id] = src_pretrained_embedding[word]
                            count_a += 1
                    sess.run(src_embed_a.assign(src_value_a))
                    sess.run(dst_embed_a.assign(dst_value_a))
                    # print(sess.run(src_embed_a))


                    count_b=0
                    src_value_b = sess.run(src_embed_b)
                    dst_value_b = sess.run(dst_embed_b)
                    for word in dst_pretrained_embedding:
                        if word in du.dst2idx:
                            id = du.dst2idx[word]
                            # print(id)
                            src_value_b[id] = dst_pretrained_embedding[word]
                            dst_value_b[id] = dst_pretrained_embedding[word]
                            count_b += 1
                    sess.run(src_embed_b.assign(src_value_b))
                    sess.run(dst_embed_b.assign(dst_value_b))

                    logger.info('restore %d src_embedding and %d dst_embedding done' %(count_a, count_b))

                except:
                    logger.info('Failed to load the pretriaed embeddings')

            # tmp_writer = codecs.open('tmp_test', 'w', 'utf-8')

            for epoch in range(1, config.train.num_epochs+1):
                for batch in du.get_training_batches_with_buckets():
                    # swap the batch[0] and batch[1] accroding to whether the length of the sequence is odd or even
                    # batch_swap=[]
                    # swap_0 = np.arange(batch[0].shape[1])
                    # swap_1 = np.arange(batch[1].shape[1])
                    #
                    # if len(swap_0) % 2 == 0:
                    #     swap_0[0::2]+=1
                    #     swap_0[1::2]-=1
                    # else:
                    #     swap_0[0:-1:2]+=1
                    #     swap_0[1::2]-=1
                    #
                    # if len(swap_1) % 2 == 0:
                    #     swap_1[0::2]+=1
                    #     swap_1[1::2]-=1
                    # else:
                    #     swap_1[0:-1:2] += 1
                    #     swap_1[1::2] -= 1
                    #
                    # batch_swap.append(batch[0].transpose()[swap_0].transpose())
                    # batch_swap.append(batch[1].transpose()[swap_1].transpose())

                    # print(batch[0])
                    # print(batch_swap[0])

                    # randomly shuffle the batch[0] and batch[1]
                    #batch_shuffle=[]
                    #shuffle_0_indices = np.random.permutation(np.arange(batch[0].shape[1]))
                    #shuffle_1_indices = np.random.permutation(np.arange(batch[1].shape[1]))
                    #batch_shuffle.append(batch[0].transpose()[shuffle_0_indices].transpose())
                    #batch_shuffle.append(batch[1].transpose()[shuffle_1_indices].transpose())


                    def get_shuffle_k_indices(length, shuffle_k):
                        shuffle_k_indices = []
                        rand_start = np.random.randint(shuffle_k)

                        indices_list_start = list(np.random.permutation(np.arange(0, rand_start)))
                        shuffle_k_indices.extend(indices_list_start)

                        for i in range(rand_start, length, shuffle_k):
                            if i + shuffle_k > length:
                                indices_list_i = list(np.random.permutation(np.arange(i, length)))
                            else:
                                indices_list_i = list(np.random.permutation(np.arange(i, i + shuffle_k)))

                            shuffle_k_indices.extend(indices_list_i)

                        return np.array(shuffle_k_indices)

                    batch_shuffle=[]
                    shuffle_0_indices = get_shuffle_k_indices(batch[0].shape[1], config.train.shuffle_k)
                    shuffle_1_indices = get_shuffle_k_indices(batch[1].shape[1], config.train.shuffle_k)
                    #print(shuffle_0_indices)
                    batch_shuffle.append(batch[0].transpose()[shuffle_0_indices].transpose())
                    batch_shuffle.append(batch[1].transpose()[shuffle_1_indices].transpose())

                    start_time = time.time()
                    step = sess.run(model.global_step)

                    step, lr, gnorm_aa, loss_aa, acc_aa, _ = sess.run(
                        [model.global_step, model.learning_rate, model.grads_norm_aa,
                         model.loss_aa, model.acc_aa, model.train_op_aa],
                        feed_dict={model.src_a_pl: batch_shuffle[0], model.dst_a_pl: batch[0]})

                    step, lr, gnorm_bb, loss_bb, acc_bb, _ = sess.run(
                        [model.global_step, model.learning_rate, model.grads_norm_bb,
                         model.loss_bb, model.acc_bb, model.train_op_bb],
                        feed_dict={model.src_b_pl: batch_shuffle[1], model.dst_b_pl: batch[1]})


                    # this step takes too much time
                    generate_ab, generate_ba = sess.run(
                        [model.generate_ab, model.generate_ba],
                        feed_dict={model.src_a_pl: batch[0], model.src_b_pl: batch[1]})

                    generate_ab_dealed, _ = deal_generated_samples(generate_ab, du.dst2idx)
                    generate_ba_dealed, _ = deal_generated_samples(generate_ba, du.src2idx)

                    #for sent in du.indices_to_words(batch[0], o='src'):
                    #    print(sent, file=tmp_writer)
                    #for sent in du.indices_to_words(generate_ab_dealed, o='dst'):
                    #    print(sent, file=tmp_writer)

                    step, acc_ab, loss_ab, _ = sess.run(
                        [model.global_step, model.acc_ab, model.loss_ab, model.train_op_ab],
                        feed_dict={model.src_a_pl:generate_ba_dealed, model.dst_b_pl: batch[1]})

                    step, acc_ba, loss_ba, _ = sess.run(
                        [model.global_step, model.acc_ba, model.loss_ba, model.train_op_ba],
                        feed_dict={model.src_b_pl:generate_ab_dealed, model.dst_a_pl: batch[0]})

                    if step % config.train.disp_freq == 0:
                        logger.info('epoch: {0}\tstep: {1}\tlr: {2:.6f}\tgnorm: {3:.4f}\tloss: {4:.4f}'
                                    '\tacc: {5:.4f}\tcross_loss: {6:.4f}\tcross_acc: {7:.4f}\ttime: {8:.4f}'
                                    .format(epoch, step, lr, gnorm_aa, loss_aa, acc_aa, loss_ab, acc_ab,
                                            time.time() - start_time))

                    # Save model
                    if step % config.train.save_freq == 0:
                        mp = config.train.logdir + '/model_epoch_%d_step_%d' % (epoch, step)
                        saver.save(sess, mp)
                        logger.info('Save model in %s.' % mp)

            logger.info("Finish training.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.train.logdir):
        os.makedirs(config.train.logdir)
    logging.basicConfig(filename=config.train.logdir+'/train.log', level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    # Train
    train(config)
