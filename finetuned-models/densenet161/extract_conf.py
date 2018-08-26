
import torch
torch.cuda.set_device(0)

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from scene_classification import Scene_Classification
from models import densessd
from models import densenet
from focal import FocalLoss

DATA_ROOT = '/home/wuxiaodong/ai_challenge/data/'
log = './log/log_log.txt'
with open(log, 'w') as f:
    f.write('\n')

nlabels = 80
display = 20

maxepoch = 40
base_lr = 0.1
gamma = 1  # each epoch learning_rate = learning_rate * gamma
state_dict = torch.load('./snapshots/ckpt_epoch_48.pytorch')




net = densenet.densenet161(state_dict)
net.cuda()



mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
crop_size = 224  
train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomSizedCrop(crop_size), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.CenterCrop(crop_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

# train_data = Scene_Classification(DATA_ROOT, train='train', transform=train_transform, useSSD=True, resize=256)
val_data = Scene_Classification(DATA_ROOT, train='val', transform=test_transform, useSSD=True, resize=256)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=24, shuffle=True,
#                                            num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False,
                                          num_workers=4, pin_memory=True)

state = {}
state['epoch'] = 0
state['test_accuracy'] = 0.0
state['test_accuracy_t3'] = 0.0




# def test():
net.eval()
max_probs = []
loss_avg = 0.0
correct = 0
correct_t3 = 0
for batch_idx, (data, target, otherfea) in enumerate(val_loader):
    print(batch_idx)
    data = torch.autograd.Variable(data.cuda())
    target = torch.autograd.Variable(target.cuda())
    # otherfea = torch.autograd.Variable(otherfea.cuda())

    output = net(data)

    output = F.softmax(output)


    pred = output.data.max(1)[1]
    topk = output.data.topk(3, dim=1)[1]
    N = target.data.shape[0]
    target_reshape = target.data.view(N, 1)
    target_enlarge = torch.cat((target_reshape, target_reshape, target_reshape), 1)

    correct += pred.eq(target.data).sum()
    correct_t3 += topk.eq(target_enlarge).sum()
    maxprob = output.data.max(1)[0]
    maxprob = list(maxprob)
    max_probs.extend(maxprob)


state['test_accuracy'] = correct / len(val_loader.dataset)
state['test_accuracy_t3'] = correct_t3 / len(val_loader.dataset)
# return max_probs


# max_probs = test()


with open('val_max_probs.txt', 'w') as f:
    for prob in max_probs:
        f.write(str(prob)+'\n')

import numpy as np

x = [0 for a in max_probs if a>0.2 else 1]
