import os
import random
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn

from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC,JointTraining,SynapticIntelligence,CoPE

def set_seed(seed=13, verbose=True):
    if verbose:
        print("Set Seed:", seed)
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_method(model, optimizer, args, eval_plugin, plugin_list, dataset_info, device):
    loss = nn.CrossEntropyLoss()
    current_mode = 'offline'
    num_instance_each_class = dataset_info['num_instance_each_class']
    num_classes = dataset_info['num_classes']
    if True:
        data_count=int(num_classes*num_instance_each_class) if current_mode=='online' else int(num_classes*num_instance_each_class*(1-0.3))
        print('data_count is {}'.format(data_count))
        data_count=min(args.max_memory_size,data_count) # buffer_size cannot be greater than 3000
        if(args.method.split("_")[-1].isnumeric()==False):
            buffer_size=data_count
        else:
            buffer_size=int(args.method.split("_")[-1])


    if args.method =='CWRStar':
        cl_strategy = CWRStar(
            model, optimizer,
            loss,cwr_layer_name=None, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list)
    elif 'Replay' in args.method: 
        cl_strategy = Replay(
            model, optimizer,
            loss, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,mem_size=buffer_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list)
    elif (args.method =='JointTraining' and current_mode=='offline'):
        cl_strategy = JointTraining(
            model, optimizer,
            loss, train_mb_size=args.batch_size, train_epochs=args.num_epoch*args.timestamp//3, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list)
    elif 'GDumbFinetune' in args.method: # TODO
        cl_strategy = GDumb(
            model, optimizer,
            loss, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=False,buffer='class_balance')
    # stanard gdumb= reset model+ class_balance buffer'
    elif 'GDumb' in args.method:
        cl_strategy = GDumb(
            model, optimizer,
            loss, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size)
    elif 'BiasReservoir' in args.method: # TODO
        if('reset' in args.method):
            resett=True
        else:
            resett=False
        alpha_mode ='Dynamic' if 'Dynamic' in args.method else 'Fixed'
        alpha_value=float(args.method.split("_")[-1])
        cl_strategy = GDumb(
            model, optimizer,
            loss, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=resett,buffer='bias_reservoir_sampling',
            alpha_mode=alpha_mode,alpha_value=alpha_value)
    # this is basically the 'reservoir sampling in the paper(no reset+ reservoir sampling'
    elif 'Reservoir' in args.method:
        cl_strategy = GDumb(
            model, optimizer,
            loss, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list,mem_size=buffer_size,reset=False,buffer='reservoir_sampling')
    elif 'Cumulative' in args.method:
        if('reset' in args.method):
            resett=True
        else:
            resett=False
        cl_strategy = Cumulative(
            model, optimizer,
            loss, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list,reset=resett)
    elif args.method =='LwF':
        cl_strategy = LwF(
            model, optimizer,
            loss,
            alpha= np.linspace(0,2,num=args.timestamp).tolist(),temperature=1,
            train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list)
    elif args.method =='GEM':
        cl_strategy = GEM(
            model, optimizer,
            loss, patterns_per_exp=data_count,memory_strength=0.5, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list)
    elif 'AGEMFixed' in args.method:
        cl_strategy = AGEM(
            model, optimizer,
            loss,patterns_per_exp=buffer_size,sample_size=buffer_size, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list,reservoir=True)
    elif 'AGEM' in args.method:
        cl_strategy = AGEM(
            model, optimizer,
            loss,patterns_per_exp=buffer_size,sample_size=buffer_size, train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list) #,reservoir=False)
    elif args.method == 'EWC':
        cl_strategy = EWC(
            model, optimizer, loss,
            ewc_lambda=0.4, mode='online',decay_factor=0.1,
            train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list)
    elif args.method == 'Naive':
        cl_strategy = Naive(
            model, optimizer, loss,
            train_mb_size=args.batch_size,             train_epochs=args.num_epoch,
            eval_mb_size=args.batch_size,             evaluator=eval_plugin,
            device=device,             plugins=plugin_list, )
    elif args.method == 'SynapticIntelligence':
        cl_strategy = SynapticIntelligence(
            model, optimizer, loss,
            si_lambda=0.0001,train_mb_size=args.batch_size, train_epochs=args.num_epoch,
            eval_mb_size=args.batch_size, evaluator=eval_plugin,device=device,plugins=plugin_list)
    elif 'CoPE' in args.method:
        cl_strategy = CoPE(
            model, optimizer, loss,
            train_mb_size=args.batch_size, train_epochs=args.num_epoch, eval_mb_size=args.batch_size,mem_size=buffer_size,
            evaluator=eval_plugin,device=device,plugins=plugin_list)
    else:
        raise NotImplementedError
    print(cl_strategy)
    return cl_strategy


