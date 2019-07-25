import numpy as np
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

import torchvision.transforms as transforms
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                
                
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
                   
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()
            
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x,axis=1,keepdims=True)
    return softmax_x 
    
def get_RANK(query_semantic, test_mask, classes):
    query_semantic = query_semantic
    test_mask = test_mask
    query_semantic = query_semantic/np.linalg.norm(query_semantic,2,axis=1,keepdims=True)
    test_mask = test_mask/np.linalg.norm(test_mask,2,axis=1,keepdims=True)
    dist = np.dot(query_semantic, test_mask.transpose())
    return classes[np.argmax(dist, axis=1)]

def compute_domain_accuracy(predict_label, domain):
    num = predict_label.shape[0]
    n = 0
    for i in predict_label:
        if i in domain:
            n +=1
            
    return float(n)/num

def compute_class_accuracy_total( true_label, predict_label, classes):
    nclass = len(classes)
    acc_per_class = np.zeros((nclass, 1))
    for i, class_i in enumerate(classes):
        idx = np.where(true_label == class_i)[0]
        acc_per_class[i] = (sum(true_label[idx] == predict_label[idx])*1.0 / len(idx))
    return np.mean(acc_per_class)

def compute_accuracy_S( true_label, rank_pre, cls_pre, classes):
    predict_label = rank_pre
    for idx,class_i in enumerate(predict_label):
        if class_i in classes:
            predict_label[idx] = cls_pre[idx]
    
    nclass = len(classes)
    acc_per_class = np.zeros((nclass, 1))
    for i, class_i in enumerate(classes):
        idx = np.where(true_label == class_i)[0]
        acc_per_class[i] = (sum(true_label[idx] == predict_label[idx])*1.0 / len(idx))
    return np.mean(acc_per_class)
        
def opt_domain_acc(cls_s,cls_t):
    ''' source domain '''
    opt_acc_s = 0
    num_s = cls_s.shape[0]
    max_score_s = np.max(cls_s,axis=1)   
          
    opt_acc_t = 0
    num_t = cls_t.shape[0]
    max_score_t = np.max(cls_t,axis=1)
    
    max_H = 0
    opt_tau = 0
    for step in range(10):
        tau = 0.1*step
        
        idx = np.where(max_score_s>tau)
        acc_s = float(idx[0].shape[0])/num_s 
        
        idx = np.where(max_score_t<tau)
        acc_t = float(idx[0].shape[0])/num_t
         
        H = 2*acc_s*acc_t/(acc_s+acc_t) 
        if H>max_H:
            opt_acc_t = acc_t
            opt_acc_s = acc_s
            max_H = H
            opt_tau = tau
    return opt_acc_s,opt_acc_t,opt_tau
            
                          
def opt_pre(rank_ss,rank_tt,score_s,score_t,gt_s, gt_t, seen_c,unseen_c):
    max_score_s = np.max(score_s,axis=1)   
    pre_cls_s = np.argmax(score_s,axis=1)
    max_score_t = np.max(score_t,axis=1)
    pre_cls_t = np.argmax(score_t,axis=1)
    
    opt_S = 0
    opt_U = 0
    opt_H = 0
    opt_tau = 0
    
    for step in range(9):
        tau = 0.1*step+0.1
        pre_s = pre_cls_s
        pre_t = pre_cls_t
    
        # seen domain
        idx = np.where(max_score_s<tau);idx = idx[0]
        pre_s[idx] = unseen_c[0]
        
        # unseen domain
        idx = np.where(max_score_t<tau);idx = idx[0]
        pre_t[idx] = rank_tt[idx]
        
        S = compute_class_accuracy_total(gt_s, pre_s,seen_c)
        U = compute_class_accuracy_total(gt_t, pre_t,unseen_c)
        H = 2*S*U/(S+U) 
         
        if H>opt_H:
            opt_S = S
            opt_U = U
            opt_H = H
            opt_tau = tau
            
    return opt_H,opt_S,opt_U,opt_tau

def preprocess_strategy(dataset):
    evaluate_transforms = None
    if 0:
        train_transforms = transforms.Compose([
            transforms.GetRandomCrop(256,224),
        ])
        val_transforms = transforms.Compose([
            transforms.GetTenCrop(256,224),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(448),
            #transforms.Resize(448),
            #transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])  
        val_transforms = transforms.Compose([
            transforms.Resize(480),
            #transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ]) 
        #evaluate_transforms = transforms.Compose([
        #    transforms.Resize(480),
        #    CenterCropWithFlip(448),
        #    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        #    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        #])
    return train_transforms, val_transforms#, evaluate_transforms


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
