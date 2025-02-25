import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
from lib.augment import degree_drop_weights, drop_edge_weighted, drop_feature
from lib.metric import balanced_accuracy_score,balanced_f1
from sklearn.metrics import confusion_matrix
from lib.balanced_cluster import balanced_kmean
from lib.kmeans_gpu import kmeans

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(train_loader,model, loss_supcon, loss_ce_class, optimizer, epoch,device, args, flag='train'):
    losses = []
    model.train()
    print('Epoch', epoch)
    y_true = np.array([])
    y_pred = np.array([])
    for idx, (graph, cluster_target) in enumerate(train_loader):
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        batch = graph.batch.to(device)
        target = graph.y.to(device)
        cluster_target = torch.tensor(cluster_target).to(device)
        
        drop_weights = degree_drop_weights(edge_index)
        edge_index1 = drop_edge_weighted(edge_index, drop_weights, p = 0.15, threshold = 0.7)
        edge_index2 = drop_edge_weighted(edge_index, drop_weights, p = 0.1, threshold = 0.7)

        bsz = target.shape[0]
        
        f1 = model.model_drug(x, edge_index1,batch)
        f2 = model.model_drug(x, edge_index2,batch)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim = 1)
        
        f1_normalize = F.normalize(f1,dim=1)
        f2_normalize = F.normalize(f2,dim=1)

        features_normalize = torch.cat([f1_normalize.unsqueeze(1), f2_normalize.unsqueeze(1)], dim = 1)

        if cluster_target[0] != -1:
            loss_scl = loss_supcon(features_normalize, target, cluster_target)  
        else:
            loss_scl = loss_supcon(features_normalize, target)
        
        if epoch > 10:
            output = model.classify(f1)
            pred = output.argmax(dim = 1) 
            y_pred = np.concatenate((y_pred, pred.to("cpu").numpy()))
            y_true = np.concatenate((y_true, target.to("cpu").numpy()))
            loss_classify = loss_ce_class(output, target)
            loss = loss_classify + loss_scl
        else:
            loss = loss_scl
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    
    if epoch > 10:
        acc = balanced_accuracy_score(y_true = y_true, y_pred = y_pred)
        print("train acc: {:.4f},  train loss: {:.5f}".format(acc, np.mean(losses)))
    else:
        print("train loss: {:.5f}".format(losses.avg))

    return np.mean(losses)

def valid(valid_loader, model, loss_ce_class, optimizer, epoch, device, args, flag='valid'):
    model.eval()
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for i, (graph, cluster_target) in enumerate(valid_loader):
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)
            target = graph.y.to(device)
            cluster_target = torch.tensor(cluster_target).to(device)
            bsz = target.shape[0]
            output = model(x,edge_index, batch)
            loss = loss_ce_class(output, target)
            
            pred = output.argmax(dim = 1) 
            y_pred=np.concatenate((y_pred, pred.to("cpu").numpy()))
            y_true=np.concatenate((y_true, target.to("cpu").numpy()))
            losses.append(loss.item())

        cf = confusion_matrix(y_true=y_true, y_pred=y_pred).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        print("valid loss: {:.4f}".format(np.mean(losses)))
        acc = balanced_accuracy_score(y_true = y_true, y_pred = y_pred)
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(out_cls_acc)
       
    return acc

def test(test_loader,model, device, flag='test'):
    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    with torch.no_grad():
        for i, (graph, cluster_target) in enumerate(test_loader):
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)
            target = graph.y.to(device)
            cluster_target = torch.tensor(cluster_target).to(device)
            output = model(x, edge_index, batch)

            pred = output.argmax(dim = 1) 
            y_pred = np.concatenate((y_pred, pred.to("cpu").numpy()))
            y_true = np.concatenate((y_true, target.to("cpu").numpy()))
        
        cf = confusion_matrix(y_true = y_true, y_pred = y_pred).astype(float)
        cls_cnt = cf.sum(axis = 1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt

        acc = balanced_accuracy_score(y_true = y_true, y_pred = y_pred)
        b_f1 = balanced_f1(y_true = y_true, y_pred = y_pred)
        print("test acc: {:.4f}, test f1: {:.4f}".format(acc, b_f1))
        out_cls_acc = '%s Class Accuracy: %s'%(flag, (np.array2string(cls_acc, separator = ',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(out_cls_acc)
       
    return acc

def cluster(train_loader_cluster, model, cluster_number, cls_num_list, args):
    model.eval()
    device = "cuda"
    features_sum = []
    for idx, (graph,cluster_target) in enumerate(train_loader_cluster):
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        batch = graph.batch.to(device)
        target = graph.y.to(device)
        cluster_target = torch.tensor(cluster_target).to(device)
        with torch.no_grad():
            features  = model.model_drug(x,edge_index,batch)
            features = features.to("cuda")
            features = F.normalize(features,dim=1)
            features = features.detach()
            features_sum.append(features)
    features= torch.cat(features_sum,dim=0)
    features = torch.split(features, cls_num_list, dim=0)
    feature_center = [torch.mean(t, dim=0) for t in features]
    feature_center = torch.cat(feature_center, axis = 0)
    feature_center = feature_center.reshape(args.num_classes, args.mlp_input_dim)
    density = np.zeros(len(cluster_number))
    for i in range(len(cluster_number)):  
        center_distance = F.pairwise_distance(features[i], feature_center[i], p=2).mean()/np.log(len(features[i])+10) 
        density[i] = center_distance.cpu().numpy()
    density = density.clip(np.percentile(density,20), np.percentile(density,80)) 
    density = args.temperature*np.exp(density/density.mean())
    for index, value in enumerate(cluster_number):
        if value==1:
            density[index] = 0.1
    target = [[] for i in range(len(cluster_number))]
    for i in range(len(cluster_number)):  
        if cluster_number[i] >1:
          if 1 ==2:
            cluster_ids_x, _ = balanced_kmean(X=features[i], num_clusters=cluster_number[i], distance='cosine', init='k-means++',iol=50,tol=1e-3,device=torch.device("cuda"))
          else:
            cluster_ids_x, _ = kmeans(X=features[i], num_clusters=cluster_number[i], distance='cosine', tol=1e-3, iter_limit=35, device=torch.device("cuda"))
            #run faster for cluster
          target[i]=cluster_ids_x
        else:
            target[i] = torch.zeros(1,features[i].size()[0], dtype=torch.int).squeeze(0)
    cluster_number_sum=[sum(cluster_number[:i]) for i in range(len(cluster_number))]
    for i, k in enumerate(cluster_number_sum):
         target[i] =  torch.add(target[i], k)
    targets = torch.cat(target, dim=0)
    targets = targets.numpy().tolist()

    return targets, density

    
def distance_class(train_loader_cluster, model, args):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_num = args.num_classes
    cluster_num = max(train_loader_cluster.dataset.new_labels) + 1

    class_sum = torch.zeros((class_num, args.mlp_input_dim)).to(device)
    class_center = torch.zeros((class_num, args.mlp_input_dim)).to(device)
    class_counts = torch.zeros(class_num)

    cluster_sum = torch.zeros((cluster_num, args.mlp_input_dim)).to(device)
    cluster_center = torch.zeros((cluster_num, args.mlp_input_dim)).to(device)
    cluster_counts = torch.zeros(cluster_num)
    category_labels = torch.zeros(cluster_num)
    with torch.no_grad():
        for i, (graph, cluster_target) in enumerate(train_loader_cluster):
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            batch = graph.batch.to(device)
            target = graph.y.to(device)
            cluster_target = torch.tensor(cluster_target).to(device)
            feature = model.model_drug(x,edge_index, batch)
            feature = F.normalize(feature, dim=1)
            for c in range(class_num):
                class_indices = (target == c).nonzero(as_tuple = True)[0]
                class_sum[c, ] += feature[class_indices].sum(dim=0)
                class_counts[c] += len(class_indices)
            for index in range(len(target)):
                category_labels[cluster_target[index]] = target[index]
            for c in range(cluster_num):
                cluster_indices = (cluster_target == c).nonzero(as_tuple = True)[0]
                cluster_sum[c, ] += feature[cluster_indices].sum(dim = 0)
                cluster_counts[c] += len(cluster_indices)

    for c in range(class_num):
        class_center[c, ] = class_sum[c] / class_counts[c]

    class_distances = []
    for i in range(class_center.shape[0]):
        min_distance = float('inf')
        for j in range(class_center.shape[0]):
            if i != j:
                class_distance = torch.norm(class_center[i] - class_center[j])
                if class_distance < min_distance:
                    min_distance = class_distance
        class_distances.append(min_distance)
    
    for c in range(cluster_num):
        cluster_center[c, ] = cluster_sum[c]/cluster_counts[c]
    cluster_distances = []
    for i in range(args.num_classes):
        class_samples = cluster_center[category_labels == i]
        min_distances = []
        for j in range(args.num_classes):
            if j != i:
                other_class_samples = cluster_center[category_labels == j]
                distances = []
                for sample in class_samples:
                    for other_sample in other_class_samples:
                        distance = torch.norm(sample - other_sample)
                        distances.append(distance)
                min_distances.append(torch.min(torch.tensor(distances)))
        cluster_distances.append(torch.min(torch.tensor(min_distances)))
    
    return torch.nn.functional.softmax(-torch.tensor(class_distances)), torch.nn.functional.softmax(-torch.tensor(cluster_distances))