
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
from truncated_loss import TruncatedLoss

DATA_ROOT = '/home/wuxiaodong/ai_challenge/data/'
log = './log/log18.txt'
with open(log, 'w') as f:
    f.write('\n')

nlabels = 80
display = 20




maxepoch = 40
base_lr = 0.001
gamma = 1  # each epoch learning_rate = learning_rate * gamma
# state_dict = torch.load('./snapshots/dense_plus_sssd_epoch_39.pytorch')
state_dict = torch.load('./snapshots/ckpt_epoch_48.pytorch')
# state_dict = torch.load('./snapshots/dense_sssd_cnn0_sgd_epoch_2fc_step3_4.pytorch')




net = densenet.densenet161(state_dict)
# net = densessd.densenet161(state_dict)
net.cuda()



mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
crop_size = 224  
train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomSizedCrop(crop_size), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.CenterCrop(crop_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

train_data = Scene_Classification(DATA_ROOT, train='train', transform=train_transform, useSSD=True, resize=256)
val_data = Scene_Classification(DATA_ROOT, train='val', transform=test_transform, useSSD=True, resize=256)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=24, shuffle=True,
                                           num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False,
                                          num_workers=4, pin_memory=True)


# optimizer = torch.optim.SGD([{'params': net.features.parameters(), 'lr': base_lr},
#     {'params': net.classifier.parameters(), 'lr': base_lr}, {'params':net.fc1.parameters(), 'lr':base_lr}], momentum=0.9,
#                             weight_decay=0.0005, nesterov=True)
optimizer = torch.optim.SGD([{'params': net.features.parameters(), 'lr': base_lr},
    {'params': net.classifier.parameters(), 'lr': base_lr}], momentum=0.5,
                            weight_decay=0.0005, nesterov=True)

# optimizer_adam = torch.optim.Adam([{'params': net.features.parameters(), 'lr': base_lr},
#     {'params': net.classifier.parameters(), 'lr': base_lr}, {'params':net.fc1.parameters(), 'lr':base_lr}])
state = {}
state['epoch'] = 0
state['test_accuracy'] = 0.0
state['test_accuracy_t3'] = 0.0

# this block is used for testing the code  
# net.train()
# data, target, otherfea = next(iter(train_loader))
# data = torch.autograd.Variable(data.cuda())
# target = torch.autograd.Variable(target.cuda())
# otherfea = torch.autograd.Variable(otherfea.cuda())

# # fc1,out = net(data)
# # optimizer_adam.zero_grad()
# one_hot = torch.rand(target.size(0),nlabels).zero_()
# one_hot.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
# one_hot = torch.autograd.Variable(one_hot.cuda())
# output = net(data, otherfea)
# criterion = nn.L1Loss()
# loss = criterion(output, one_hot)
# loss.backward()
# optimizer_adam.step()



def train():
    net.train()
    loss_avg = 0.0
    for batch_idx, (data, target, otherfea) in enumerate(train_loader):
        data = torch.autograd.Variable(data.cuda())
        target = torch.autograd.Variable(target.cuda())
        otherfea = torch.autograd.Variable(otherfea.cuda())

        # target = target.view(target.size()[0],1)

        # forward
        output = net(data)
        # output = net(data, otherfea)

        #backward
        optimizer.zero_grad()
        # optimizer_adam.zero_grad()

        #-------------------------------- mseloss-------------------------------
        # criterion = nn.L1Loss()
        # one_hot = torch.rand(target.size(0),nlabels).zero_()
        # one_hot.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
        # one_hot = torch.autograd.Variable(one_hot.cuda())
        # loss = criterion(output, one_hot)
        #------------------------------- mseloss---------------------------------

        #------------------------------- cross entropy loss---------------------------------
        # loss = F.cross_entropy(output, target)
        #------------------------------- cross entropy loss---------------------------------
        # focal_module = FocalLoss(gamma=6)
        # loss = focal_module(output, target)

        turncated_loss_module = TruncatedLoss()
        loss = turncated_loss_module(output, target)

        loss.backward()
        optimizer.step()
        # optimizer_adam.step()


        # calculate the loss moving average
        loss_avg = loss_avg * 0.2 + loss.data[0] * 0.8
        if batch_idx % display == 0:
            toprint = 'epoch: {}, batch id: {}, loss: {}, avg_loss: {}, accuracy: {}, acccyracy_t3: {}'.format(state['epoch'], batch_idx, round(loss.data[0], 5), round(loss_avg, 5), round(state['test_accuracy'],3), round(state['test_accuracy_t3'],3))
            print(toprint)
            with open(log, 'a') as f:
                f.write(toprint+'\n')
        state['train_loss'] = loss_avg



def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    correct_t3 = 0
    for batch_idx, (data, target, otherfea) in enumerate(val_loader):
        data = torch.autograd.Variable(data.cuda())
        target = torch.autograd.Variable(target.cuda())
        otherfea = torch.autograd.Variable(otherfea.cuda())

        output = net(data)
        # output = net(data, otherfea)

        #-------------------------------- l1loss-------------------------------
        # criterion = nn.L1Loss()
        # one_hot = torch.rand(target.size(0),nlabels).zero_()
        # one_hot.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
        # one_hot = torch.autograd.Variable(one_hot.cuda())
        # loss = criterion(output, one_hot)
        #------------------------------- l1loss---------------------------------

        #------------------------------- cross entropy loss---------------------------------
        # loss = F.cross_entropy(output, target)
        #------------------------------- cross entropy loss---------------------------------
        # focal_module = FocalLoss(gamma=6)
        # loss = focal_module(output, target)


        turncated_loss_module = TruncatedLoss()
        loss = turncated_loss_module(output, target)

        pred = output.data.max(1)[1]
        topk = output.data.topk(3, dim=1)[1]
        N = target.data.shape[0]
        target_reshape = target.data.view(N, 1)
        target_enlarge = torch.cat((target_reshape, target_reshape, target_reshape), 1)

        correct += pred.eq(target.data).sum()
        correct_t3 += topk.eq(target_enlarge).sum()

        # loss_avg += loss.data[0]

    # state['test_loss'] = loss_avg / len(val_loader)
    state['test_accuracy'] = correct / len(val_loader.dataset)
    state['test_accuracy_t3'] = correct_t3 / len(val_loader.dataset)


for epoch in range(maxepoch):
    state['epoch'] = epoch
    print('Epoch {} '.format(epoch))
    base_lr = base_lr * gamma
    # if epoch==5:
    #     base_lr = 0.1 * base_lr
    # if epoch==13:
    #     base_lr = 0.1 * base_lr
    optimizer.param_groups[1]['lr'] = base_lr
    optimizer.param_groups[0]['lr'] = base_lr 
    train()
    test()
    torch.save(net.state_dict(), './snapshots/densenet_truncated_step0_{}.pytorch'.format(epoch))


















# val_data = Scene_Classification(DATA_ROOT, train='val', useSSD=True, resize=256)
# nlabels = 8
# val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

# # for batch_idx, (data, target, ssdfea)  in enumerate(val_loader):
# #     pass

# data, target, ssdfea = next(iter(val_loader))
# data = Variable(data)
# target = Variable(target)
# ssdfea = Variable(ssdfea)
# # input = Variable(torch.randn(1,3,224,224))

# model = densessd.DenseNet()

# # features = model(input)



# features = model.features(data)

# out = F.relu(features, inplace=True)
# out = F.avg_pool2d(out, kernel_size=model.avgpool_size).view(
#                    features.size(0), -1)
# ssdfea = ssdfea.view(features.size(0),-1)
# addedfea = torch.cat((out, ssdfea), 1)
# norm_model = nn.BatchNorm2d(addedfea.size(1))
# out = norm_model(addedfea)
# out = model.classifier(out)