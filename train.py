import torch
import torch.optim as optim
import numpy as np
from Net.DAMIDL import DAMIDL
from DataLoader.Data_Loader import data_flow, tst_data_flow
import scipy.io as sio
from sklearn.metrics import roc_curve, auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]

task = 1

epoch = 100
learning_rate = 0.001
batch_size = 10
patch_size = 25
patch_num = 60
kernel_num = [32, 64, 128, 128]

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
sample_name = data['samples_train'].flatten()
labels = data['labels_train'].flatten()

res = np.zeros(shape=(5, 4))

# 5 fold
for i in range(5):
    # 20% training samples as the validation set
    valid_list = range(len(sample_name)//5*i, len(sample_name)//5*(i+1))
    train_list = list(set(range(len(sample_name))).difference(set(valid_list)))
    labels_train = labels[train_list]
    labels_valid = labels[valid_list]
    samples_train = sample_name[train_list]
    samples_valid = sample_name[valid_list]

    # load patch location proposals calculated on training samples
    template_cors = sio.loadmat('SelectPatches/'
                                'template_center_fold{}_size{}_num{}.mat'.format(i+1, patch_size, patch_num))
    template_cors = template_cors['patch_centers']

    # build model
    model = DAMIDL(patch_num=patch_num, feature_depth=kernel_num)
    model = model.to(device).cuda()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pos_num = np.sum(labels_valid == 1)
    neg_num = np.sum(labels_valid == 0)

    best = 0

    # train
    for e in range(epoch):
        train_loader = data_flow(img_path, samples_train, labels_train, template_cors,
                                 batch_size, patch_size, patch_num)
        model.train()
        l = []
        a = []
        for i_batch in range(len(train_list)//batch_size):
            inputs, outputs = next(train_loader)
            inputs = inputs.to(device).cuda()
            subject_outputs = torch.from_numpy(outputs).long().flatten().to(device).cuda()

            model.zero_grad()
            optimizer.zero_grad()

            subject_pred = model(inputs)
            loss = criterion(subject_pred, subject_outputs)

            loss.backward()
            optimizer.step()

            l.append(loss.item())
            subject_pred = subject_pred.max(1)[1]

            acc = torch.sum(torch.eq(subject_pred, subject_outputs)).cpu().numpy()
            a.append(acc/batch_size)
            print('[fold{} epoch{} batch{} train] cur_acc:{} mean_acc:{}'
                  ' cur_loss:{} mean_loss:{}'.format(i+1, e, i_batch, a[-1], np.mean(a), l[-1], np.mean(l)))

        # validation
        acc = 0
        model.eval()
        TP = 0
        TN = 0
        subject_probs = []
        for i_batch in range(len(samples_valid)):
            inputs, outputs = tst_data_flow(img_path, samples_valid[i_batch], labels_valid[i_batch],
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

        acc = acc / len(valid_list)
        sen = TP / pos_num
        spe = TN / neg_num
        fpr, tpr, thresholds = roc_curve(labels_valid, subject_probs)
        roc_auc = auc(fpr, tpr)

        if best < acc:
            best = acc
            res[i][0] = best
            res[i][1] = sen
            res[i][2] = spe
            res[i][3] = roc_auc
            torch.save(model, 'results/{}/best_model_fold{}.pkl'.format(task_name, i + 1))

        print('[fold{} epoch{} validation] cur_acc:{} best_acc:{}'.format(i + 1, e, acc, best))


print(res)
sio.savemat('results/{}/validation_performance.mat'.format(task_name), {'results': res})





