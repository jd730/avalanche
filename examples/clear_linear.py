################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 05-06-2022                                                             #
# Author: Jia Shi, Zhiqiu Lin                                                  #
# E-mail: jiashi@andrew.cmu.edu, zl279@cornell.edu                             #
# Website: https://clear-benchmark.github.io                                   #
################################################################################

"""
Example: Training and evaluating on CLEAR benchmark (pre-trained features)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    confusion_matrix_metrics,
    disk_usage_metrics,
)
from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC,JointTraining,SynapticIntelligence,CoPE
from avalanche.benchmarks.classic.clear import CLEAR, CLEARMetric

from clear_utils import get_method

# For CLEAR dataset setup
#DATASET_NAME = "clear100_cvpr2022"
#DATASET_NAME = "clear100"
DATASET_NAME = "clear10_neurips2021"

DATA_INFO = {
        'clear10_neurips2021': {
            'num_classes': 11,
            'num_instance_each_class' : 300,
            'num_instance_each_class_test' : 150
            },
        'clear10': {
            'num_classes': 11,
            'num_instance_each_class' : 300,
            'num_instance_each_class_test' : 150
            }
        }



CLEAR_FEATURE_TYPE = "moco_b0"  # MoCo V2 pretrained on bucket 0
# CLEAR_FEATURE_TYPE = "moco_imagenet"  # MoCo V2 pretrained on imagenet
# CLEAR_FEATURE_TYPE = "byol_imagenet"  # BYOL pretrained on imagenet
# CLEAR_FEATURE_TYPE = "imagenet"  # Pretrained Imagenet model

# please refer to paper for discussion on streaming v.s. iid protocol
EVALUATION_PROTOCOL = "streaming"  # trainset = testset per timestamp
# EVALUATION_PROTOCOL = "iid"  # 7:3 trainset_size:testset_size

# For saving the datasets/models/results/log files
ROOT = Path("CLEAR")
DATA_ROOT = ROOT / DATASET_NAME
MODEL_ROOT = ROOT / "models"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# Define hyperparameters/model/scheduler/augmentation

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CLEARDataset Training')

    parser.add_argument('-input_dim', type=int, default=2048)
    parser.add_argument('-output_dim', type=int, default=512)
    parser.add_argument('-method', type=str, default='Naive')
    parser.add_argument('-dataset', type=str, default='clear10_neurips2021', choices=['clear10_neurips2021', 'clear100_cvpr2022', 'clear10', 'clear100'])
    parser.add_argument('-batch_size', type=int, default=1024)
    parser.add_argument('-num_epoch', type=int, default=1)
    parser.add_argument('-timestamp', type=int, default=10)
    parser.add_argument('-max_memory_size', type=int, default=3000000)

    # LR
    parser.add_argument('-start_lr', type=float, default=1)
    parser.add_argument('-scheduler_step', type=float, default=0.1)
    parser.add_argument('-step_scheduler_decay', type=int, default=60)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-momentum', type=float, default=0.9)


    args = parser.parse_args()
    return args

def main(args):
    # feature size is 2048 for resnet50
    num_classes = DATA_INFO[args.dataset]['num_classes']
    model = torch.nn.Linear(args.input_dim, num_classes)
    print(model)

    def make_scheduler(optimizer, step_size, gamma=0.1):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        return scheduler

    # log to Tensorboard
    tb_logger = TensorboardLogger(ROOT)

    # log to text file
    text_logger = TextLogger(open(ROOT / "log.txt", "w+"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True,
                         stream=True),
#        accuracy_metrics(minibatch=True, epoch=True, experience=True,
#                         stream=True, trained_experience=True), 
        loss_metrics(minibatch=True, epoch=True, experience=True,
                     stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(
            num_classes=num_classes, save_image=False,
            stream=True
        ),
        disk_usage_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loggers=[interactive_logger, text_logger, tb_logger],
    )

    if EVALUATION_PROTOCOL == "streaming":
        seed = None
    else:
        seed = 0

    benchmark = CLEAR(
        data_name = DATASET_NAME,
        evaluation_protocol=EVALUATION_PROTOCOL,
        feature_type=CLEAR_FEATURE_TYPE,
        seed=seed,
        dataset_root=DATA_ROOT,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.start_lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = make_scheduler(
        optimizer, args.step_scheduler_decay, args.scheduler_step)

    plugin_list = [LRSchedulerPlugin(scheduler)]
    cl_strategy = get_method(model, optimizer, args, eval_plugin, plugin_list, DATA_INFO[args.dataset], device)
    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    print("Current protocol : ", EVALUATION_PROTOCOL)
    for index, experience in enumerate(benchmark.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        res = cl_strategy.train(experience)
        torch.save(
            model.state_dict(),
            str(MODEL_ROOT / f"model{str(index).zfill(2)}.pth")
        )
        print("Training completed")
        print(
            "Computing accuracy on the whole test set with"
            f" {EVALUATION_PROTOCOL} evaluation protocol"
        )
        results.append(cl_strategy.eval(benchmark.test_stream))
    # generate accuracy matrix
    num_timestamp = len(results)
    accuracy_matrix = np.zeros((num_timestamp, num_timestamp))
    for train_idx in range(num_timestamp):
        for test_idx in range(num_timestamp):
            mname = f'Top1_Acc_Exp/eval_phase/test_stream/Task00{test_idx}/Exp00{test_idx}'
            accuracy_matrix[train_idx][test_idx] = results[train_idx][mname]
    print("Accuracy_matrix : ")
    print(accuracy_matrix)
    metric = CLEARMetric().get_metrics(accuracy_matrix)
    print(metric)

    metric_log = open(ROOT / "metric_log.txt", "w+")
    metric_log.write(
        f"Protocol: {EVALUATION_PROTOCOL} "
        f"Seed: {seed} "
        f"Feature: {CLEAR_FEATURE_TYPE} \n"
    )
    json.dump(accuracy_matrix.tolist(), metric_log, indent=6)
    json.dump(metric, metric_log, indent=6)
    metric_log.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
