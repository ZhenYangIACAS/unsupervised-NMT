from evaluate import Evaluator
import yaml
import time
import numpy as np
import os
import sys
import logging
from share_function import FlushFile
from utils import DataUtil, AttrDict

evaluate_config=sys.argv[1]
config=AttrDict(yaml.load(open(evaluate_config)))
evaluator=Evaluator(config)


idx=0
bleu=0.10
logFile='DevResult'

sys.stdout=FlushFile(sys.stdout)
logging.basicConfig(level=logging.INFO,
                   filename=logFile,
                   filemode='a')

if not os.path.isdir('./model_best'):
    os.system('mkdir model_best')
else:
    logging.info("model_best dir exists")

while True:
    idx +=1
    logging.info('idx: '+str(idx))
    evaluator.saver.restore(evaluator.sess, config.test.modelFile)
    evaluator.evaluate_translate_only()
    command_line ="./multi-bleu.perl "+config.test.ori_dst_path+" < "+config.test.output_path
    fd = os.popen(command_line)
    output=fd.read()
    print(output)
    try:
        BLEUStrIdx = output.index('BLEU = ')
        bleu_new = float(output[BLEUStrIdx+7:BLEUStrIdx+12])
        if bleu_new > bleu:
            bleu=bleu_new
            command_line="cp "+config.test.modelFile+".*"+ " ./model_best/"
            os.popen(command_line)
            logging.info('best update in idex '+str(idx))

        logging.info('bleu: '+str(bleu_new)+'\n')
    except:
        logging.info('BLEU score not found!')

    fd.close()
    time.sleep(5000)


