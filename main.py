import argparse
import json
import os
import random
import csv
from typing import List

import numpy as np
import torch

from module.trainer import LearningEnv
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:36"#新增的

#设置随机种子
def set_random_seed(seed: int):#77
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='This code is for ECPE task.')

    # Training Environment
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--num_process', default=int(os.cpu_count() * 0.8), type=int)
    parser.add_argument('--num_worker', default=6, type=int)
    parser.add_argument('--port', default=1234, type=int)

    parser.add_argument('--model_name', default='PRG_MoE')
    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--test', default=False)

    parser.add_argument('--split_directory', default=None)
    parser.add_argument('--train_data', default="data/data_fold/data_0/dailydialog_train.json")
    parser.add_argument('--valid_data', default="data/data_fold/data_0/dailydialog_valid.json")
    parser.add_argument('--test_data', default="data/data_fold/data_0/dailydialog_test.json")
    parser.add_argument('--log_directory', default='logs', type=str)
    parser.add_argument('--data_label', help='the label that attaches to saved model', default='dailydialog_fold_0')

    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--n_speaker', help='the number of speakers', default=2, type=int)
    parser.add_argument('--n_emotion', help='the number of emotions', default=7, type=int)
    parser.add_argument('--n_cause', help='the number of causes', default=2, type=int)
    parser.add_argument('--n_expert', help='the number of causes', default=4, type=int)
    parser.add_argument('--guiding_lambda', help='the mixing ratio', default=0.6, type=float)

    parser.add_argument('--max_seq_len', help='the max length of each tokenized utterance', default=75, type=int)
    parser.add_argument('--contain_context', help='While tokenizing, previous utterances are contained or not', default=False)

    parser.add_argument('--training_iter', default=40, type=int)
    parser.add_argument('--batch_size', default=1, type=int)#之前是5
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--patience', help='patience for Early Stopping', default=None, type=int)

    return parser.parse_args()

#确保预训练模型已经加载
def test_preconditions(args: argparse.Namespace):
    if args.test:
        assert args.pretrained_model is not None, "For test, you should load pretrained model."


def main():
    args = parse_args()

    test_preconditions(args)

    set_random_seed(77)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in args.gpus])
    # for i in vars(args):
    #     print("*********",i)
    #把上面的参数完全传入
    trainer = LearningEnv(**vars(args))
    trainer.run(**vars(args))

if __name__ == "__main__":
    main()
