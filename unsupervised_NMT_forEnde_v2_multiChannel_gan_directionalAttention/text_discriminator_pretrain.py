from __future__ import print_function
import yaml
import time
import os
import sys
import logging
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model
from cnn_text_discriminator import text_DisCNN
from share_function import deal_generated_samples
from share_function import extend_sentence_to_maxlen
from share_function import FlushFile

def gan_train(config):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    default_graph=tf.Graph()
    with default_graph.as_default():
        
        sess = tf.Session(config=sess_config, graph=default_graph)
        logger = logging.getLogger('')

        dis_filter_sizes = [i for i in range(1, config.train.dis_max_len, 4)]
        dis_num_filters = [(100+i * 10) for i in range(1, config.train.dis_max_len, 4)]

        #print("the scope is ", config.train.dis_scope)
       
        discriminator = text_DisCNN(
            sess=sess,
            max_len=config.train.dis_max_len,
            num_classes=3,
            vocab_size_s=config.src_vocab_size_a,
            vocab_size_t=config.dst_vocab_size_b,
            batch_size=config.train.dis_batch_size,
            dim_word=config.train.dis_dim_word,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            source_dict=config.train.dis_src_vocab,
            target_dict=config.train.dis_dst_vocab,
            gpu_device=config.train.devices,
            s_domain_data=config.train.s_domain_data,
            t_domain_data=config.train.t_domain_data,
            s_domain_generated_data=config.train.s_domain_generated_data,
            t_domain_generated_data=config.train.t_domain_generated_data,
            dev_s_domain_data=config.train.dev_s_domain_data,
            dev_t_domain_data=config.train.dev_t_domain_data,
            dev_s_domain_generated_data=config.train.dev_s_domain_generated_data,
            dev_t_domain_generated_data=config.train.dev_t_domain_generated_data,
            max_epoches=config.train.dis_max_epoches,
            dispFreq=config.train.dis_dispFreq,
            saveFreq=config.train.dis_saveFreq,
            saveto=config.train.dis_saveto,
            reload=config.train.dis_reload,
            clip_c=config.train.dis_clip_c,
            optimizer=config.train.dis_optimizer,
            reshuffle=config.train.dis_reshuffle,
            scope=config.train.text_scope
        )

        logging.info("text_discriminator pretrain begins!")
        discriminator.train()
        logging.info("text_discriminator pretrain done")


if __name__ == '__main__':
    sys.stdout = FlushFile(sys.stdout)
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
    gan_train(config)

