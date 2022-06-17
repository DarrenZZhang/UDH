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
if len(sys.argv) > 2:
    code_length = int(sys.argv[1])
    fixed_var = float(sys.argv[2])
else:
    code_length = 24
    fixed_var = 3
training_epoch = 3
max_iter = 50
num_samples = 2000
gamma = 200
sampling_times = 10
ignore_bit_percentage = ['10', '20', '30', '40', '50', '60', '70'] # 
if code_length == 48:
    ignore_bit_percentage = ['10', '20', '60', '70', '80', '90']

def calc_sim(train_label, database_label):
    with torch.no_grad():
        S = (train_label.mm(database_label.t()) > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))
        r = S.sum() / (1-S).sum()
        S = S*(1+r) - r
        return S
        
def arctanh(x):
    return 0.5 * torch.log((1+x) / (1-x+1e-10) + 1e-10)
    
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

def forward_distributional_model(model, images, code_length, mean_type):
    output = model(images)
    mean = output[:,:code_length]
    var = output[:, code_length:]
    var = torch.nn.functional.softplus(var) + 1e-3
    if mean_type == 'raw':
        mean = mean
    elif mean_type == 'tanh':
        mean = torch.tanh(mean)
    elif mean_type == 'signed':
        mean = torch.sign(mean)
    else:
        raise Exception('unsupported mean_type (accepted type: raw, tanh, sign)')
        
    return mean, var      
    
def generate_valid_bit_mask(mean, var, ignore_bit_percentage):

    code_length = mean.shape[1]
    valid_mask = torch.ones(mean.size(), dtype=mean.dtype, layout=mean.layout, device=mean.device)
    
    negative_risk = mean.abs() - 3 * var
    thre = torch.sort(negative_risk, dim=1)[0][:,int(ignore_bit_percentage * code_length)]
    
    valid_mask[negative_risk <= thre[:,None]] = 0
    return valid_mask
    
# cifar 10
database = DatasetProcessingCIFAR_10('cifar10_database_set.txt', transform=transform_train)
testset = DatasetProcessingCIFAR_10('cifar10_test_set.txt', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=1)

nclasses = 10
test_labels = LoadLabel('cifar10_test_set.txt')
test_labels_onehot = EncodingOnehot(test_labels, nclasses).cuda()
data_labels = LoadLabel('cifar10_database_set.txt')
data_labels_onehot = EncodingOnehot(data_labels, nclasses).cuda()

num_test, num_data =  len(testset), len(database)

# model
model = models.alexnet(pretrained=True)
model.classifier[6]=nn.Linear(4096, code_length * 2)
print(model)
model = model.cuda()

# optimizer
if code_length == 12:
    lr = 1e-3
elif code_length == 24 or code_length == 32 or code_length == 48:
    lr = 1e-4
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
            mean, var = forward_distributional_model(model, images, code_length=code_length, mean_type='raw')
            signed_mean = torch.tanh(mean)
            
            U[u_ind, :] = signed_mean.data
            S = sim[u_ind, :]
            
            mean_loss = (signed_mean.mm(V.t()) - code_length * S).pow(2).sum() + gamma * (V[batch_ind, :] - signed_mean).pow(2).sum()
            mean_loss = mean_loss / num_data / batch_size_
            sampled_means = []
            for _ in range(sampling_times):
                sampled_means.append(torch.tanh(torch.normal(mean, var)))
            sampled_means = torch.stack(sampled_means, dim=0)   # [sampling_times, N, bit]
            sampled_means_loss = (torch.matmul(sampled_means, V.t()) - code_length * S[None,:,:]).pow(2).sum() + gamma * (V[None, batch_ind, :] - sampled_means).pow(2).sum()
            sampled_means_loss = sampled_means_loss / num_data / batch_size_ / sampling_times
            
            #fixed_var = 0.5
            num_sampled_x = 50
            sampled_x = torch.linspace(-0.5, 3, num_sampled_x)[None, None, :].cuda().detach()
            sampled_x = sampled_x * torch.sign(mean)[:,:,None].data.float()
            sampled_Fx_p = 1/(var[:,:,None]*2.51) * torch.exp(-0.5 * ((sampled_x - mean[:,:,None]) / var[:,:,None]).pow(2) )
            sampled_prior_p = (2*fixed_var+1) * torch.tanh(sampled_x).pow(2).pow(fixed_var) * (1-torch.tanh(sampled_x).pow(2))
            zero_prob_bit = (torch.sign(sampled_x) == torch.sign(mean)[:,:,None].data).float()
            sampled_prior_p = sampled_prior_p * zero_prob_bit
            sampled_means_dis_loss = sampled_Fx_p * torch.log(sampled_Fx_p / (sampled_prior_p+1e-10) + 1e-10)
            sampled_means_dis_loss = sampled_means_dis_loss.sum() / batch_size_ / num_sampled_x
            
            
            optimizer.zero_grad()
            loss = mean_loss + 0.3*sampled_means_loss + 0.1*sampled_means_dis_loss
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
        valid_bit_mask = dict()
        for i in ignore_bit_percentage:
            valid_bit_mask[i] = torch.zeros([num_test, code_length]).cuda() 
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels, batch_ind = data
                images = images.cuda()
                mean, var = forward_distributional_model(model, images, code_length=code_length, mean_type='raw')
                mean = mean.squeeze()
                T[batch_ind,:] = torch.sign(mean)
                
                for i in ignore_bit_percentage:
                    valid_bit_mask[i][batch_ind,:] = generate_valid_bit_mask(mean, var, float(i) / 100.0)
                
            map = CalcHR.CalcMap_cuda(T, V, test_labels_onehot, data_labels_onehot)
            top5b_map = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, topk=500)
            top1k_map = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, topk=1000)
            top5k_map = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, topk=5000)
            rmap = CalcHR.CalcMap_cuda_R(T, V, test_labels_onehot, data_labels_onehot, R=2)
            print('[%d] MAP:%.4f, MAP@top500:%.4f, MAP@top1k:%.4f, MAP@top5k:%.4f, MAP@2:%.4f' %(iter+1, map, top5b_map, top1k_map, top5k_map, rmap))
            
            for i in ignore_bit_percentage:
                map_with_ignore = CalcHR.CalcMap_cuda(T, V, test_labels_onehot, data_labels_onehot, valid_bit_mask=valid_bit_mask[i])
                top5b_map_with_ignore = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, valid_bit_mask=valid_bit_mask[i], topk=500)
                top1k_map_with_ignore = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, valid_bit_mask=valid_bit_mask[i], topk=1000)
                top5k_map_with_ignore = CalcHR.CalcMap_cuda_topk(T, V, test_labels_onehot, data_labels_onehot, valid_bit_mask=valid_bit_mask[i], topk=5000)
                rmap_with_ignore = CalcHR.CalcMap_cuda_R(T, V, test_labels_onehot, data_labels_onehot, valid_bit_mask=valid_bit_mask[i], R=2)
                print('[%d] MAP:%.4f, MAP@top500:%.4f, MAP@top1k:%.4f, MAP@top5k:%.4f, MAP@2:%.4f, w/ %d%% ignore rate' %(iter+1, map_with_ignore, top5b_map_with_ignore, top1k_map_with_ignore, top5k_map_with_ignore, rmap_with_ignore, int(i)))
    
    scheduler.step()

    print('epoch time spend: [%d]s' % (time.time() - epoch_timer))
