import os

train_anno = './cifar10_origin/train.txt'
test_anno = './cifar10_origin/val.txt'

# train set
for line in open(train_anno, 'r'):
    line = line.strip()
    batch_id = line.split('/')[0][-1]
    label = line.split(' ')[-1]
    image_name = line.split(' ')[0].split('/')[-1]
    os.system('cp ./cifar10_origin/batch{}/{} ./cifar10/train/{}_{}_{}'.format(batch_id, image_name, label, batch_id, image_name))

# test set
for line in open(train_anno, 'r'):
    line = line.strip()
    label = line.split(' ')[-1]
    image_name = line.split(' ')[0].split('/')[-1]
    os.system('cp ./cifar10_origin/test/{} ./cifar10/test/{}_999_{}'.format(image_name, label, image_name))
