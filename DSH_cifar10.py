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


# TODO:
# 1. modify model structure and output
# 2. F_step_criterion

# args
batch_size = 48
code_length = 64
training_epoch = 50


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
model.classifier[6]=nn.Linear(4096, code_length)
print(model)
model = model.cuda()

# optimizer
optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-4, eps=1e-3)
#scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.1,
#                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
scheduler = MultiStepLR(optimizer, milestones=[15,30,45], gamma=0.1)

# hash 
B = torch.randn(num_train, code_length).cuda()
H = torch.zeros(num_train, code_length).cuda()
B = torch.sign(torch.sign(B))
Y = train_labels_onehot
D = Y.t().mm(Y) + torch.eye(nclasses).cuda()
D = D.inverse()
D = D.mm(Y.t())
#F_step_criterion = torch.nn.MSELoss(reduction='sum')


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

        output = model(images)
        
        H[batch_ind, :] = output.data
        #loss = F_step_criterion(output, temp) / batch_size
        loss = (B[batch_ind, :] - output).pow(2).sum() / batch_size
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()
    print('[%d] loss: %.4f' % (epoch + 1, running_loss / len(trainloader) * batch_size) )

    #t1 = time.time()
    #print(t1-epoch_timer)

    # eval
    if (epoch > 15) or ((epoch + 1) % 5 == 0):
        T = torch.zeros([num_test, code_length]).cuda()
        H_B = torch.zeros([num_data, code_length]).cuda()
        T_C = torch.zeros([num_test, code_length]).cuda()
        H_C = torch.zeros([num_data, code_length]).cuda()
        model.eval()
        with torch.no_grad():
        
            for data in databaseloader:
                images, labels, batch_ind = data
                images = images.cuda()
                outputs = model(images)
                outputs = outputs.squeeze()
                H_B[batch_ind,:] = torch.sign(outputs)
                H_C[batch_ind,:] = outputs
    
            #t2 = time.time()
            #print(t2-t1)

            for data in testloader:
                images, labels, batch_ind = data
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                outputs = outputs.squeeze()
                T[batch_ind,:] = torch.sign(outputs)
                T_C[batch_ind,:] = outputs
    
            #t3 = time.time()
            #print(t3-t2)
        
            map = CalcHR.CalcMap_cuda(T, H_B, test_labels_onehot, data_labels_onehot)
            rmap = CalcHR.CalcMap_cuda_R(T, H_B, test_labels_onehot, data_labels_onehot)
            rerank_rmap = CalcHR.CalcMap_cuda_R_rerank(T, H_B, T_C, H_C, test_labels_onehot, data_labels_onehot)
            
            #map = CalcHR.CalcMap(T.cpu().data.numpy(), H_B.cpu().data.numpy(), test_labels_onehot.cpu().data.numpy(), data_labels_onehot.cpu().data.numpy())
            print('[%d] MAP:%.4f, MAP@2:%.4f, rerank-MAP@2:%.4f' %(epoch+1, map, rmap, rerank_rmap))

            #scheduler.step(map)
            
    scheduler.step()

        #print(time.time()-t3)
    print('epoch time spend: [%d]s' % (time.time() - epoch_timer))
