import os
import numpy as np
import scipy.io as sio
from random import sample

task = 1
img_path = "../ADNI/"
sample_name = []
labels = []


if task == 1:
    for img in os.listdir(img_path + 'AD'):
        sample_name.append('AD/' + img)
        labels.append(1)
    for img in os.listdir(img_path + 'NC'):
        sample_name.append('NC/' + img)
        labels.append(0)
    task_name = 'AD_classification'
elif task == 2:
    for img in os.listdir(img_path + 'pMCI'):
        sample_name.append('pMCI/' + img)
        labels.append(1)
    for img in os.listdir(img_path + 'sMCI'):
        sample_name.append('sMCI/' + img)
        labels.append(0)
    task_name = 'MCI_conversion'
elif task == 3:
    for img in os.listdir(img_path + 'pMCI'):
        sample_name.append('pMCI/' + img)
        labels.append(1)
    for img in os.listdir(img_path + 'NC'):
        sample_name.append('NC/' + img)
        labels.append(0)
    task_name = 'pMCI_NC'
elif task == 4:
    for img in os.listdir(img_path + 'sMCI'):
        sample_name.append('sMCI/' + img)
        labels.append(1)
    for img in os.listdir(img_path + 'NC'):
        sample_name.append('NC/' + img)
        labels.append(0)
    task_name = 'sMCI_NC'

sample_name = np.array(sample_name)
labels = np.array(labels)
permut = np.random.permutation(len(sample_name))
np.take(sample_name, permut, out=sample_name)
np.take(labels, permut, out=labels)

pos_list = np.arange(len(labels))[labels == 1]
neg_list = np.arange(len(labels))[labels == 0]

pos_test = sample(list(pos_list), round(len(pos_list)/5))
neg_test = sample(list(neg_list), round(len(neg_list)/5))

test_list = pos_test + neg_test
test_list = sorted(test_list)
train_list = list(set(range(len(sample_name))).difference(set(test_list)))

samples_train = sample_name[train_list]
labels_train = labels[train_list]
samples_test = sample_name[test_list]
labels_test = labels[test_list]

sio.savemat('data_split/{}/data.mat'.format(task_name), {"samples_train": samples_train,
                                                         "samples_test": samples_test,
                                                         "labels_train": labels_train,
                                                         "labels_test": labels_test})
