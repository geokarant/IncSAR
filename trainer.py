import sys
import logging
import copy
import torch
#print(torch.cuda.is_available())
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters,soft_voting
import os
import csv
import numpy as np
import time
import random
import itertools
def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    acc_tasks = []
    avg_accs = []
    args["device"] = device
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        if args["late_fusion"]== True:
            acc_task, avg_acc = _train_late_fusion(args)
        elif args["attention_fusion"] == True:
            acc_task, avg_acc = _train_attention_fusion(args)
        else:
            acc_task, avg_acc = _train(args)
    acc_tasks.append(acc_task)
    avg_accs.append(avg_acc)

def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))

        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()       
        logging.info("Top-1 Accuracy per Task: {}".format(cnn_accy["grouped"]))

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("Top-1 Accuracy curve: {}".format(cnn_curve["top1"]))
        logging.info("Average Accuracy: {:.2f} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    return cnn_curve["top1"], sum(cnn_curve["top1"])/len(cnn_curve["top1"])

def _train_late_fusion(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type_vit"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    args_vit = copy.deepcopy(args)
    args_vit["backbone_type"] = args["backbone_type_vit"]
    args_vit["finetune"] = args["finetune_vit"]
    args_vit["tuned_epoch"]= args["tuned_epoch_vit"]
    args_vit["rpca"] = args["rpca_vit"]
    args_vit["use_RP"] = args["use_RP_vit"]
    args_vit["M"] = args["M_vit"]

    data_manager_vit = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args_vit,
    )
    
    args_vit["nb_classes"] = data_manager_vit.nb_classes # update args
    args_vit["nb_tasks"] = data_manager_vit.nb_tasks
    model_vit = factory.get_model(args_vit["model_name"], args_vit)
    
    _set_random(args["seed"])
    args_cnn = copy.deepcopy(args)
    args_cnn["backbone_type"] = args["backbone_type_cnn"]
    args_cnn["finetune"] = args["finetune_cnn"]
    args_cnn["tuned_epoch"]= args["tuned_epoch_cnn"]
    args_cnn["rpca"] = args["rpca_cnn"]
    args_cnn["use_RP"] = args["use_RP_cnn"]
    args_cnn["M"] = args["M_cnn"]
    data_manager_cnn = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args_cnn,
    )
    
    args_cnn["nb_classes"] = data_manager_cnn.nb_classes # update args
    args_cnn["nb_tasks"] = data_manager_cnn.nb_tasks
    model_cnn = factory.get_model(args_cnn["model_name"], args_cnn)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager_vit.nb_tasks):
        
        logging.info("ViT params: {}".format(count_parameters(model_vit._network)))
        _set_random(args["seed"])
        model_vit.incremental_train(data_manager_vit)
        
        logits_vit, y_pred_vit, y_true= model_vit.eval_task_late_fusion()
        model_vit.after_task()

        logging.info("CNN params: {}".format(count_parameters(model_cnn._network)))
        _set_random(args["seed"])
        model_cnn.incremental_train(data_manager_cnn)
        logits_cnn, _, _= model_cnn.eval_task_late_fusion()
        model_cnn.after_task()
        cnn_accy = soft_voting(logits_vit, logits_cnn, y_true, args['init_cls'],args['increment'])

        logging.info("Top-1 Accuracy per Task:: {}".format(cnn_accy["grouped"]))

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("Top-1 Accuracy curve: {}".format(cnn_curve["top1"]))
        logging.info("Average Accuracy: {:.2f} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    return cnn_curve["top1"], sum(cnn_curve["top1"])/len(cnn_curve["top1"])

def _train_attention_fusion(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type_vit"]
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    args_vit = copy.deepcopy(args)
    args_vit["backbone_type"] = args["backbone_type_vit"]
    args_vit["rpca"] = args["rpca_vit"]
    data_manager_vit = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args_vit,
    )
    args["nb_classes"] = data_manager_vit.nb_classes # update args
    args["nb_tasks"] = data_manager_vit.nb_tasks

    args_cnn = copy.deepcopy(args)
    args_cnn["backbone_type"] = args["backbone_type_cnn"]
    args_cnn["rpca"] = args["rpca_cnn"]
    data_manager_cnn = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args_cnn,
    )

    from models.incsar_attention import Learner
    model = Learner(args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager_vit.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))

        model.incremental_train(data_manager_vit, data_manager_cnn)
        cnn_accy, nme_accy = model.eval_task_attention_fusion()
        model.after_task()

       
        logging.info("Top-1 Accuracy per Task: {}".format(cnn_accy["grouped"]))

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        cnn_curve["top1"].append(cnn_accy["top1"])

        logging.info("Top-1 Accuracy curve: {}".format(cnn_curve["top1"]))
        logging.info("Average Accuracy: {:.2f} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    return cnn_curve["top1"], sum(cnn_curve["top1"])/len(cnn_curve["top1"])

def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus

def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
