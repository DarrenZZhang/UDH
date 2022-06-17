import glob, os, random
from functools import reduce

ori_test = glob.glob('./cifar10/test/*')
ori_train = glob.glob('./cifar10/train/*')
all_data = ori_test+ori_train
random.shuffle(all_data)
categoried_all_data = [[] for _ in range(10)]

for i in all_data:
    class_id = int(i.strip().split('/')[-1].split('_')[0])
    categoried_all_data[class_id].append(i + '\n')

test_data = reduce(lambda x1, x2: x1 + x2[:100], categoried_all_data, [])
train_data = reduce(lambda x1, x2: x1 + x2[100:600], categoried_all_data, [])
database_data = reduce(lambda x1, x2: x1 + x2[100:], categoried_all_data, [])

# remove last \n
test_data[-1] = test_data[-1][:-1]
train_data[-1] = train_data[-1][:-1]
database_data[-1] = database_data[-1][:-1]

def write_file(str_list, name):
    f = open(name, 'w')
    f.writelines(str_list)
    f.close()
    
write_file(test_data, 'test_set.txt')
write_file(train_data, 'train_set.txt')
write_file(database_data, 'database_set.txt')
