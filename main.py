import argparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import time
import numpy as np
from lib.dataset import graphDataset
from LTDD_revise.lib.model import Model
import torch
import torch.nn as nn
from lib.loss import SupConLoss_rank, SupConLoss
from functools import partial
import torch.optim.lr_scheduler as lr_scheduler
from dgl.data.utils import load_graphs
from collections import Counter
from lib.adjust_lr import warmup_learning_rate,adjust_learning_rate, WarmupMultiStepLR
from lib.training import seed_everything, train, valid, test, cluster, distance_class
import os
from datetime import datetime


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gcn_feat_dim', nargs='*', default=[128,128,128], type=int, help='gcn hidden feature')
parser.add_argument('--classify_input_dim', default=128, type=int, help='classify input feature')
parser.add_argument('--classify_feat_dim', nargs='*', default=[128,128,128], type=int, help='classify hidden feature')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument( '--step', default=10, type=int,
                     help='steps for updating cluster')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gamma', default=0.99, type=float, 
                    help='gamma for schedule')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--lr_decay_rate', default=0.1, type=float,
                        help='decay rate for learning rate')
parser.add_argument('--wd', '--weight-decay', default=3e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--cluster', default=10, type=int,
                    metavar='N', help='the low limit of cluster')
parser.add_argument('--seed', default=2024, type=int,
                    help='seed for initializing training.')
parser.add_argument('--device', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--temperature', default=0.1, type=float,
                    help='softmax temperature')
parser.add_argument('--dataset', default="HIV", type=str,
                    help='dataset name')
parser.add_argument('--split', default="standard", type=str,
                    help='spplit method')
parser.add_argument('--feature_weight', default=1, type=float,
                    help='loss weight of feature part')
parser.add_argument('--scl_weight', default=0.1, type=float,
                    help='scl loss weight')
parser.add_argument('--resample', default=True, type=bool,
                    help='resample dataset')
parser.add_argument('--num_workers', default=4, type=int,
                    help='resample method')
args = parser.parse_args()

print("Prepare Data......")
seed_everything()
if args.dataset == "USPTO50k":
    args.num_classes = 10
elif args.dataset == "HIV" or args.dataset == "SBAP":
    args.num_classes = 2

dataset_name = args.dataset
split_method = args.split
dataset_dir="./data/{}/{}/{}_{}".format(dataset_name,split_method,dataset_name,split_method)
train_glist, train_label_dict = load_graphs("{}_train.bin".format(dataset_dir))
valid_glist, valid_label_dict = load_graphs("{}_valid.bin".format(dataset_dir))
test_glist, test_label_dict = load_graphs("{}_test.bin".format(dataset_dir))

sorted_pairs = sorted(zip(train_label_dict["label"], train_glist), key = lambda x: x[0])
train_label_dict_sorted, train_glist_sorted = zip(*sorted_pairs)

train_data_list=[]
for i in range(len(train_glist_sorted)):
    train_data_list.append(Data(x=train_glist_sorted[i].ndata["h"], edge_index=torch.stack((train_glist_sorted[i].edges()[0], train_glist_sorted[i].edges()[1])), y=torch.tensor([train_label_dict_sorted[i]])))   

valid_data_list=[]
for i in range(len(valid_glist)):
    valid_data_list.append(Data(x=valid_glist[i].ndata["h"], edge_index=torch.stack((valid_glist[i].edges()[0], valid_glist[i].edges()[1])), y=torch.tensor([valid_label_dict["label"][i]])))     

test_data_list=[]
for i in range(len(test_glist)):
    test_data_list.append(Data(x=test_glist[i].ndata["h"], edge_index=torch.stack((test_glist[i].edges()[0], test_glist[i].edges()[1])), y=torch.tensor([test_label_dict["label"][i]])))     

print("Contribute Dataset......")

train_dataset=graphDataset(train_data_list, args, mode="train")
valid_dataset=graphDataset(valid_data_list, args, mode="valid")
test_dataset=graphDataset(test_data_list, args, mode="test")
train_loader = DataLoader(
    train_dataset,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = args.num_workers,
    pin_memory = True,
)
train_loader_cluster = DataLoader(
    train_dataset,
    batch_size = args.batch_size,
    shuffle = False,
    num_workers = args.num_workers,
    pin_memory = True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = args.num_workers,
    pin_memory = True)
test_loader = DataLoader(
    test_dataset,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers = args.num_workers,
    pin_memory = True)
print("Start Training....")

current_time = datetime.now().strftime('%Y%m%d%H%M%S')
dir_path = f"./modelsave/{dataset_name}_{current_time}"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

num_classes = args.num_classes
cls_num_list = train_dataset.get_cls_num_list()
cluster_number = [t//max(min(cls_num_list), args.cluster) for t in cls_num_list]
for index, value in enumerate(cluster_number):
    if value == 0:
        cluster_number[index] = 1

model  = Model(gcn_hidden_dim = args.gcn_feat_dim, classify_input_dim = args.classify_input_dim, classify_hidden_dim = args.classify_feat_dim, num_classes = args.num_classes)
device="cuda"
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = args.gamma)
loss_ce_class = nn.CrossEntropyLoss()
max_valid_acc = 0
best_epoch = 0
epochs = args.epochs
for epoch in range(0, epochs):
    if epoch < 10:
        loss_supcon = SupConLoss(temperature=args.temperature).to(device)
    else:
        if epoch % args.step == 0:
            cluster_targets,density = cluster(train_loader_cluster, model, cluster_number, cls_num_list,args)
            train_dataset.new_labels = cluster_targets
            class_distance, cluster_distance = distance_class(train_loader_cluster, model,args)
            loss_ce_class = nn.CrossEntropyLoss(weight = ((class_distance+cluster_distance)/2).to(device))
        loss_supcon = SupConLoss_rank(num_class = num_classes, ranking_temperature = density).to(device)
    train_loss = train(train_loader, model, loss_supcon, loss_ce_class, optimizer, epoch, device, args)

    if epoch >=10:
        valid_acc = valid(valid_loader, model, loss_ce_class, optimizer, epoch, device, args, flag='valid')
        torch.save(model, '{}/model_{}.pth'.format(dir_path, epoch))
        if valid_acc > max_valid_acc:
            best_epoch = epoch
            torch.save(model, '{}/best_epoch.pth'.format(dir_path))
            max_valid_acc = valid_acc
        print("valid acc {:.4f},  max valid acc {:.4f}".format(valid_acc, max_valid_acc))
    if (epoch % 50 == 0) and (epoch > 49) :
        print("#########################################")
        model_best = torch.load('{}/best_epoch.pth'.format(dir_path))
        test_acc = test(test_loader, model_best, device, flag='test')
        print("#########################################")
    scheduler.step()