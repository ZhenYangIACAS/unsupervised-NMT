from __future__ import print_function
import yaml
import time
import os
import sys
import numpy as np
import logging
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model
from cnn_text_discriminator import text_DisCNN
from share_function import deal_generated_samples
from share_function import deal_generated_samples_to_maxlen
from share_function import extend_sentence_to_maxlen
from share_function import prepare_gan_dis_data
from share_function import FlushFile

def gan_train(config):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    default_graph=tf.Graph()
    with default_graph.as_default():
        sess = tf.Session(config=sess_config, graph=default_graph)

        logger = logging.getLogger('')
        du = DataUtil(config=config)
        du.load_vocab(src_vocab=config.generator.src_vocab,
                      dst_vocab=config.generator.dst_vocab,
                      src_vocab_size=config.src_vocab_size_a,
                      dst_vocab_size=config.src_vocab_size_b)

        generator = Model(config=config, graph=default_graph, sess=sess)
        generator.build_variational_train_model()

        generator.init_and_restore(modelFile=config.generator.modelFile)

        dis_filter_sizes = [i for i in range(1, config.discriminator.dis_max_len, 4)]
        dis_num_filters = [(100 + i * 10) for i in range(1, config.discriminator.dis_max_len, 4)]

        discriminator = text_DisCNN(
            sess=sess,
            max_len=config.discriminator.dis_max_len,
            num_classes=3,
            vocab_size_s=config.dst_vocab_size_a,
            vocab_size_t=config.dst_vocab_size_b,
            batch_size=config.discriminator.dis_batch_size,
            dim_word=config.discriminator.dis_dim_word,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            source_dict=config.discriminator.dis_src_vocab,
            target_dict=config.discriminator.dis_dst_vocab,
            gpu_device=config.discriminator.dis_gpu_devices,
            s_domain_data=config.discriminator.s_domain_data,
            t_domain_data=config.discriminator.t_domain_data,
            s_domain_generated_data=config.discriminator.s_domain_generated_data,
            t_domain_generated_data=config.discriminator.t_domain_generated_data,
            dev_s_domain_data=config.discriminator.dev_s_domain_data,
            dev_t_domain_data=config.discriminator.dev_t_domain_data,
            dev_s_domain_generated_data=config.discriminator.dev_s_domain_generated_data,
            dev_t_domain_generated_data=config.discriminator.dev_t_domain_generated_data,
            max_epoches=config.discriminator.dis_max_epoches,
            dispFreq=config.discriminator.dis_dispFreq,
            saveFreq=config.discriminator.dis_saveFreq,
            saveto=config.discriminator.dis_saveto,
            reload=config.discriminator.dis_reload,
            clip_c=config.discriminator.dis_clip_c,
            optimizer=config.discriminator.dis_optimizer,
            reshuffle=config.discriminator.dis_reshuffle,
            scope=config.discriminator.dis_scope
        )

        batch_iter = du.get_training_batches(
            set_train_src_path=config.generator.src_path,
            set_train_dst_path=config.generator.dst_path,
            set_batch_size=config.generator.batch_size,
            set_max_length=config.generator.max_length
        )

        for epoch in range(1, config.gan_iter_num + 1):
            for gen_iter in range(config.gan_gen_iter_num):
                batch = next(batch_iter)
                x, y = batch[0], batch[1]
                generate_ab, generate_ba = generator.generate_step(x, y)

                logging.info("generate the samples")
                generate_ab_dealed, generate_ab_mask = deal_generated_samples(generate_ab, du.dst2idx)
                generate_ba_dealed, generate_ba_mask = deal_generated_samples(generate_ba, du.src2idx)

                
                ## for debug
                #print('the generate_ba_dealed is ')
                #sample_str=du.indices_to_words(generate_ba_dealed, 'src')
                #print(sample_str)

                #print('the generate_ab_dealed is ')
                #sample_str=du.indices_to_words(generate_ab_dealed, 'dst')
                #print(sample_str)
                

                x_to_maxlen = extend_sentence_to_maxlen(x)
                y_to_maxlen = extend_sentence_to_maxlen(y)

                logging.info("calculate the reward")
                rewards_ab = generator.get_reward(x=x,
                                               x_to_maxlen=x_to_maxlen,
                                               y_sample=generate_ab_dealed,
                                               y_sample_mask=generate_ab_mask,
                                               rollnum=config.rollnum,
                                               disc=discriminator,
                                               max_len=config.discriminator.dis_max_len,
                                               bias_num=config.bias_num,
                                               data_util=du,
                                               direction='ab')

                rewards_ba = generator.get_reward(x=y,
                                               x_to_maxlen=y_to_maxlen,
                                               y_sample=generate_ba_dealed,
                                               y_sample_mask=generate_ba_mask,
                                               rollnum=config.rollnum,
                                               disc=discriminator,
                                               max_len=config.discriminator.dis_max_len,
                                               bias_num=config.bias_num,
                                               data_util=du,
                                               direction='ba')
                

                loss_ab = generator.generate_step_and_update(x, generate_ab_dealed, rewards_ab)

                loss_ba = generator.generate_step_and_update(y, generate_ba_dealed, rewards_ba)

                print("the reward for ab and ba is ", rewards_ab, rewards_ba)
                print("the loss is for ab and ba is", loss_ab, loss_ba)

                logging.info("save the model into %s" % config.generator.modelFile)
                generator.saver.save(generator.sess, config.generator.modelFile)


            ####  modified to here, next starts from here

            logging.info("prepare the gan_dis_data begin")
            data_num = prepare_gan_dis_data(
                train_data_source=config.generator.src_path,
                train_data_target=config.generator.dst_path,
                gan_dis_source_data=config.discriminator.s_domain_data,
                gan_dis_positive_data=config.discriminator.t_domain_data,
                num=config.generate_num,
                reshuf=True
            )
            
            s_domain_data_half = config.discriminator.s_domain_data+'.half'
            t_domain_data_half = config.discriminator.t_domain_data+'.half'

            os.popen('head -n ' + str(config.generate_num / 2) + ' ' + config.discriminator.s_domain_data + ' > ' + s_domain_data_half)
            os.popen('tail -n ' + str(config.generate_num / 2) + ' ' + config.discriminator.t_domain_data + ' > ' + t_domain_data_half)
            
            logging.info("generate and the save t_domain_generated_data in to %s." %config.discriminator.s_domain_generated_data)

            generator.generate_and_save(data_util=du,
                                        infile=s_domain_data_half,
                                        generate_batch=config.discriminator.dis_batch_size,
                                        outfile=config.discriminator.t_domain_generated_data,
                                        direction='ab'
                                      )

            logging.info("generate and the save s_domain_generated_data in to %s." %config.discriminator.t_domain_generated_data)

            generator.generate_and_save(data_util=du,
                                        infile=t_domain_data_half,
                                        generate_batch=config.discriminator.dis_batch_size,
                                        outfile=config.discriminator.s_domain_generated_data,
                                        direction='ba'
                                      )
            
            logging.info("prepare %d gan_dis_data done!" %data_num)
            logging.info("finetuen the discriminator begin")

            discriminator.train(max_epoch=config.gan_dis_iter_num,
                                s_domain_data=config.discriminator.s_domain_data,
                                t_domain_data=config.discriminator.t_domain_data,
                                s_domain_generated_data=config.discriminator.s_domain_generated_data,
                                t_domain_generated_data=config.discriminator.t_domain_generated_data
                                )
            discriminator.saver.save(discriminator.sess, discriminator.saveto)
            logging.info("finetune the discrimiantor done!")

        logging.info('reinforcement training done!')

if __name__ == '__main__':
    sys.stdout = FlushFile(sys.stdout)
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.logdir):
        os.makedirs(config.logdir)
    logging.basicConfig(filename=config.logdir+'/train.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    # Train
    gan_train(config)

