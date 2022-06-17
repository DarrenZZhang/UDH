import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class DatasetProcessingCIFAR_10(Dataset):
    def __init__(self,  img_filename, transform=None):
        
        self.transform = transform
        # reading img file from file
        img_filepath = img_filename
        fp = open(img_filepath, 'r')

        # print(x.strip().split(' ')[1] for x in fp)
        # self.img_filename = [x.strip().split(' ')[0] for x in fp]
        # self.labels = [int(x.strip().split(' ')[1]) for x in fp]
        
        self.img_filename = []
        self.labels = []
        for x in fp:
            image = './'+x.strip()
            self.img_filename.append(image)
            label = x.strip().split('/')[3].split('_')[0]
            self.labels.append(int(label))
        fp.close()

    def __getitem__(self, index):
        img = Image.open(self.img_filename[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.labels[index]])
        return img, label, index
    def __len__(self):
        return len(self.img_filename)


import torch.utils.data.sampler as sampler

class SubsetSampler(sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
