import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import  torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import CalcHammingRanking as CalcHR
from BatchReader import DatasetProcessingCIFAR_10
import time
import numpy as np

import pdb

# args
batch_size = 48
code_length = 64
training_epoch = 50
sampling_times = 100
ignore_bit_percentage = 0.1

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


class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


transform_train = transforms.Compose([
    ResizeImage(256),
    transforms.RandomCrop(224),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])
resize_size = 256
crop_size = 224
start_center = (resize_size - crop_size - 1) / 2

transform_test = transforms.Compose([
    ResizeImage(256),
    PlaceCrop(224, start_center, start_center),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# cifar 10
trainset = DatasetProcessingCIFAR_10('cifar10_train_set.txt',transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = DatasetProcessingCIFAR_10('cifar10_test_set.txt',transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

database = DatasetProcessingCIFAR_10('cifar10_database_set.txt',transform=transform_test)
databaseloader = torch.utils.data.DataLoader(database, batch_size=batch_size * 4,
                                         shuffle=False, num_workers=4)
nclasses = 10
test_labels = LoadLabel('cifar10_test_set.txt')
test_labels_onehot = EncodingOnehot(test_labels, nclasses).cuda()
train_labels = LoadLabel('cifar10_train_set.txt')
train_labels_onehot = EncodingOnehot(train_labels, nclasses).cuda()

data_labels = LoadLabel('cifar10_database_set.txt')
data_labels_onehot = EncodingOnehot(data_labels, nclasses).cuda()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_train, num_test, num_data =  len(trainset), len(testset), len(database)


# model
model = models.alexnet(pretrained=True)
model.classifier[6]=nn.Linear(4096, code_length * 2)    # output dim is double code_length, including mean and variance
print(model)
model = model.cuda()

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

# optimizer
optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-4, eps=1e-3)
scheduler = MultiStepLR(optimizer, milestones=[15,30,45], gamma=0.1)

# hash 
B = torch.randn(num_train, code_length).cuda()
H = torch.zeros(num_train, code_length).cuda()
B = torch.sign(torch.sign(B))
Y = train_labels_onehot
D = Y.t().mm(Y) + torch.eye(nclasses).cuda()
D = D.inverse()
D = D.mm(Y.t())


highest_rerank_rmap = -1
for epoch in range(training_epoch):
    print('start epoch [%d]...' % epoch)
    epoch_timer = time.time()

    # D-Step
    D_ = D.mm((B))
    B = torch.sign(Y.mm(D_) + 1e-5 * H)

    model.train()
    running_loss = 0.0

    # F-Step
    for i, batches in enumerate(trainloader):
        images, labels, batch_ind = batches
        
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        #output = model(images)
        mean, var = forward_distributional_model(model, images, code_length=code_length, mean_type='raw')
        sampled_means = []
        for _ in range(sampling_times):
            sampled_means.append(torch.normal(mean, var))
        sampled_means = torch.stack(sampled_means, dim=0)   # [sampling_times, N, bit]
        
        mean_loss = (B[batch_ind, :] - mean).pow(2).sum() / batch_size
        sampled_means_loss = (B[batch_ind, :].repeat(sampling_times, 1, 1) - sampled_means).pow(2).sum() / sampling_times / batch_size
        # kl of two normal distributions: (1/2)*((var1/var2)^2 + (mean1-mean2)^2/var2^2 - 1) + log(var2/var1)
        fixed_var = 1./ 3.   # 1/2, 1/3
        sampled_means_dis_loss = 0.5 * ( (var/fixed_var).pow(2) + (mean - torch.sign(mean.detach())).pow(2)/(fixed_var*fixed_var) - 1 ) + torch.log(fixed_var/var)
        sampled_means_dis_loss = sampled_means_dis_loss.sum() / batch_size
        
        H[batch_ind, :] = mean.data
        loss = mean_loss + 0.3*sampled_means_loss + 0.1*sampled_means_dis_loss
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
        
    print('[%d] loss: %.4f' % (epoch + 1, running_loss / len(trainloader) * batch_size) )

    # eval
    if (epoch > 15) or ((epoch + 1) % 5 == 0):
        T = torch.zeros([num_test, code_length]).cuda()
        H_B = torch.zeros([num_data, code_length]).cuda()
        T_C = torch.zeros([num_test, code_length]).cuda()
        H_C = torch.zeros([num_data, code_length]).cuda()
        valid_bit_mask = torch.zeros([num_test, code_length]).cuda() 
        model.eval()
        with torch.no_grad():

            
            #if epoch == 14:
            #    pdb.set_trace()
            

            for data in databaseloader:
                images, labels, batch_ind = data
                images = images.cuda()
                # outputs = model(images)
                mean, var = forward_distributional_model(model, images, code_length=code_length, mean_type='raw')
                mean = mean.squeeze()
                H_B[batch_ind,:] = torch.sign(mean)
                H_C[batch_ind,:] = mean
    
            for data in testloader:
                images, labels, batch_ind = data
                images, labels = images.cuda(), labels.cuda()
                # outputs = model(images)
                mean, var = forward_distributional_model(model, images, code_length=code_length, mean_type='raw')
                mean = mean.squeeze()
                T[batch_ind,:] = torch.sign(mean)
                T_C[batch_ind,:] = mean
                
                valid_bit_mask[batch_ind,:] = generate_valid_bit_mask(mean, var, ignore_bit_percentage)
    
            map = CalcHR.CalcMap_cuda(T, H_B, test_labels_onehot, data_labels_onehot)
            rmap = CalcHR.CalcMap_cuda_R(T, H_B, test_labels_onehot, data_labels_onehot)
            rerank_rmap = CalcHR.CalcMap_cuda_R_rerank(T, H_B, T_C, H_C, test_labels_onehot, data_labels_onehot)
            map_with_ignore = CalcHR.CalcMap_cuda(T, H_B, test_labels_onehot, data_labels_onehot, valid_bit_mask=valid_bit_mask)
            rmap_with_ignore = CalcHR.CalcMap_cuda_R(T, H_B, test_labels_onehot, data_labels_onehot, valid_bit_mask=valid_bit_mask)
            rerank_rmap_with_ignore = CalcHR.CalcMap_cuda_R_rerank(T, H_B, T_C, H_C, test_labels_onehot, data_labels_onehot, valid_bit_mask=valid_bit_mask)
            
            print('[%d] MAP:%.4f, MAP@2:%.4f, rerank-MAP@2:%.4f' %(epoch+1, map, rmap, rerank_rmap))
            print('[%d] MAP:%.4f, MAP@2:%.4f, rerank-MAP@2:%.4f, w/ %d%% ignore rate' %(epoch+1, map_with_ignore, rmap_with_ignore, rerank_rmap_with_ignore, int(ignore_bit_percentage*100)))

        if rerank_rmap > highest_rerank_rmap:
            torch.save(model, './checkpoint.pt')
            print('save model')
            highest_rerank_rmap = rerank_rmap
            
    scheduler.step()
    print('epoch time spend: [%d]s' % (time.time() - epoch_timer))
