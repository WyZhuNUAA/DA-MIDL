import torch
import numpy as np
from Net.DAMIDL import DAMIDL
from DataLoader.Data_Loader import data_flow, tst_data_flow
import scipy.io as sio
from sklearn.metrics import roc_curve, auc
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]

patch_size = 25
patch_num = 60
template_cors = sio.loadmat('SelectPatches/template_center_size{}_num{}.mat'.format(patch_size, patch_num))
template_cors = template_cors['patch_centers']


task = 1
img_path = "../ADNI/"
if task == 1:
    task_name = 'AD_classification'
elif task == 2:
    task_name = 'MCI_conversion'
elif task == 3:
    task_name = 'pMCI_NC'
elif task == 4:
    task_name = 'sMCI_NC'
data = sio.loadmat('data_split/{}/data.mat'.format(task_name))
samples_test = data['samples_test'].flatten()
labels_test = data['labels_test'].flatten()

# img_path = "../AIBL/"
# samples_test = []
# labels_test = []
# if task == 1:
#     for img in os.listdir(img_path + 'AD'):
#         samples_test.append('AD/' + img)
#         labels_test.append(1)
#     for img in os.listdir(img_path + 'NC'):
#         samples_test.append('NC/' + img)
#         labels_test.append(0)
#     task_name = 'AD_classification'
# elif task == 2:
#     for img in os.listdir(img_path + 'pMCI'):
#         samples_test.append('pMCI/' + img)
#         labels_test.append(1)
#     for img in os.listdir(img_path + 'sMCI'):
#         samples_test.append('sMCI/' + img)
#         labels_test.append(0)
#     task_name = 'MCI_conversion'
# elif task == 3:
#     for img in os.listdir(img_path + 'pMCI'):
#         samples_test.append('pMCI/' + img)
#         labels_test.append(1)
#     for img in os.listdir(img_path + 'NC'):
#         samples_test.append('NC/' + img)
#         labels_test.append(0)
#     task_name = 'pMCI_NC'
# elif task == 4:
#     for img in os.listdir(img_path + 'sMCI'):
#         samples_test.append('sMCI/' + img)
#         labels_test.append(1)
#     for img in os.listdir(img_path + 'NC'):
#         samples_test.append('NC/' + img)
#         labels_test.append(0)
#     task_name = 'sMCI_NC'
# samples_test = np.array(samples_test)
# labels_test = np.array(labels_test)


model = torch.load('results/{}/best_model.pkl'.format(task_name))
model.eval()

pos_num = np.sum(labels_test == 1)
neg_num = np.sum(labels_test == 0)
acc = 0
TP = 0
TN = 0
subject_probs = []

for i_batch in range(len(samples_test)):
    inputs, outputs = tst_data_flow(img_path, samples_test[i_batch], labels_test[i_batch],
                                    template_cors, patch_size, patch_num)
    inputs = inputs.to(device).cuda()
    subject_outputs = torch.from_numpy(outputs).long().flatten().to(device).cuda()
    subject_pred = model(inputs)
    subject_prob = subject_pred.cpu().detach().numpy()[:, 1][0]
    subject_probs.append(subject_prob)
    subject_pred = subject_pred.max(1)[1]

    if subject_outputs.cpu().numpy()[0] == 1 and subject_pred.cpu().numpy()[0] == 1:
        TP += 1
    if subject_outputs.cpu().numpy()[0] == 0 and subject_pred.cpu().numpy()[0] == 0:
        TN += 1
    acc += torch.sum(torch.eq(subject_pred, subject_outputs)).cpu().numpy()
acc = acc / len(samples_test)
sen = TP / pos_num
spe = TN / neg_num
fpr, tpr, thresholds = roc_curve(labels_test, subject_probs)
roc_auc = auc(fpr, tpr)
print('ACC:', acc)
print('SEN:', sen)
print('SPE:', spe)
print('AUC:', roc_auc)
sio.savemat('results/{}/test_performance.mat'.format(task_name), {'ACC': acc, 'SEN': sen, 'SPE': spe, 'AUC': roc_auc})
# sio.savemat('results_AIBL/{}/test_performance.mat'.format(task_name),
#             {'ACC': acc, 'SEN': sen, 'SPE': spe, 'AUC': roc_auc})


