import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import CalcHammingRanking as CalcHR
from BatchReader import DatasetProcessingCIFAR_10, SubsetSampler
import time
import sys
import numpy as np

import pdb

# args
batch_size = 64
if len(sys.argv) > 1:
    code_length = int(sys.argv[1])
else:
    code_length = 24
training_epoch = 3
max_iter = 50
num_samples = 2000
gamma = 200

def calc_sim(train_label, database_label):
    S = (train_label.mm(database_label.t()) > 0).float()
    S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S
  
def LoadLabel(filename):
    path = filename
    labels = []
    fp = open(path, 'r')
    for x in fp:
        label = x.strip().split('/')[3].split('_')[0]
        labels.append(int(label))
    fp.close()
    return torch.LongTensor(list(map(int, labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot
    
transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# cifar 10
database = DatasetProcessingCIFAR_10('cifar10_database_set.txt', transform=transform_train)
testset = DatasetProcessingCIFAR_10('cifar10_test_set.txt', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

nclasses = 10
test_labels = LoadLabel('cifar10_test_set.txt')
test_labels_onehot = EncodingOnehot(test_labels, nclasses).cuda()
data_labels = LoadLabel('cifar10_database_set.txt')
data_labels_onehot = EncodingOnehot(data_labels, nclasses).cuda()

num_test, num_data =  len(testset), len(database)

# model
model = models.alexnet(pretrained=True)
model.classifier[6]=nn.Linear(4096, code_length)
model.classifier.add_module('7', nn.Tanh())
print(model)
model = model.cuda()

# optimizer
if code_length == 32 or code_length == 24 or code_length == 12:
    lr = 1e-3
elif code_length == 48:
    lr = 1e-4
#optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, lr=lr)
#scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5,)
scheduler = ExponentialLR(optimizer, 0.9)

# hash 
V = torch.randn((num_data, code_length), dtype=torch.float32).sign().cuda() 

for iter in range(max_iter):
    print('start epoch [%d]...' % (iter + 1))
    epoch_timer = time.time()
    model.train()
    
    # sample a training set for every V update
    select_index = list(np.random.permutation(range(num_data)))[0: num_samples]
    
    sampler = SubsetSampler(select_index)
    trainloader = torch.utils.data.DataLoader(database, 
                                             batch_size=batch_size,
                                             shuffle=False, 
                                             sampler=sampler,
                                             num_workers=2)
    
    # feature learning
    sample_label = data_labels_onehot[select_index, :]
    sim = calc_sim(sample_label, data_labels_onehot)
    
    U = torch.zeros((num_samples, code_length), dtype=torch.float32).cuda()
    
    for epoch in range(training_epoch):
        for i, batches in enumerate(trainloader):
            images, labels, batch_ind = batches
            batch_size_ = labels.size()[0]
            u_ind = np.linspace(i * batch_size, np.min((num_samples, (i+1)*batch_size)) - 1, batch_size_, dtype=int)
        
            images = images.cuda()
            output = model(images)
            
            U[u_ind, :] = output.data
            S = sim[u_ind, :]
            
            loss = (output.mm(V.t()) - code_length * S).pow(2).sum() + gamma * (V[batch_ind, :] - output).pow(2).sum()
            loss = loss / num_data / batch_size_
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    # code learning
    with torch.no_grad():
        barU = torch.zeros((num_data, code_length)).cuda()
        barU[select_index, :] = U
    
        Q = code_length * sim.t().mm(U) + gamma * barU 
        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
        
            V[:, k] = torch.sign(Q[:, k] - V_.mm(U_.t().mm(Uk[:,None])).squeeze()) 
            
        loss_ = (U.mm(V.t()) - code_length * sim).pow(2).sum() + gamma * (V[select_index, :] - U).pow(2).sum()
        print('[%d] loss: %.4f' % (iter + 1, loss_.item() / num_samples / num_data))

    # eval
    if (iter > 10) or ((iter + 1) % 5 == 0):
        T = torch.zeros([num_test, code_length]).cuda()
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels, batch_ind = data
                images = images.cuda()
                outputs = model(images)
                outputs = outputs.squeeze()
                T[batch_ind,:] = torch.sign(outputs)
                
        map = CalcHR.CalcMap_cuda(T, V, test_labels_onehot, data_labels_onehot)
        top5b_map = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, topk=500)
        top1k_map = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, topk=1000)
        top5k_map = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, topk=5000)
        rmap = CalcHR.CalcMap_cuda_R(T, V, test_labels_onehot, data_labels_onehot, R=2)
            
        print('[%d] MAP:%.4f, MAP@top500:%.4f, MAP@top1k:%.4f, MAP@top5k:%.4f, MAP@2:%.4f' %(iter+1, map, top5b_map, top1k_map, top5k_map, rmap))

    scheduler.step()

    print('epoch time spend: [%d]s' % (time.time() - epoch_timer))
