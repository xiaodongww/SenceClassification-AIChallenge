import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(target, classes):
    mask = torch.rand(target.size(0),classes).zero_()
    mask.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
    mask = torch.autograd.Variable(mask.cuda())

    return mask

def truncate(input, threshold):
    threshold = -threshold
    output = -input
    nn_threshold = nn.Threshold(threshold, -1)
    output = nn_threshold(output)
    output = -output
    return output


class TruncatedLoss(nn.Module):

    def __init__(self, threshold=0.8):
        super(TruncatedLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        y = one_hot(target, input.size(1))

        logit = F.softmax(input)
        logit = truncate(logit, self.threshold)
        loss = -1 * y * torch.log(logit) # cross entropy
        return loss.sum()/float(input.size(0))


# import torch
# torch.cuda.set_device(0)

# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.autograd import Variable
# from scene_classification import Scene_Classification
# from models import densessd
# from models import densenet
# from focal import FocalLoss
# # from truncated_loss import TruncatedLoss

# DATA_ROOT = '/home/wuxiaodong/ai_challenge/data/'
# log = './log/log18.txt'
# with open(log, 'w') as f:
#     f.write('\n')

# nlabels = 80
# display = 20




# maxepoch = 40
# base_lr = 0.001
# gamma = 1  # each epoch learning_rate = learning_rate * gamma
# # state_dict = torch.load('./snapshots/dense_plus_sssd_epoch_39.pytorch')
# state_dict = torch.load('./snapshots/ckpt_epoch_48.pytorch')
# # state_dict = torch.load('./snapshots/dense_sssd_cnn0_sgd_epoch_2fc_step3_4.pytorch')




# net = densenet.densenet161(state_dict)
# # net = densessd.densenet161(state_dict)
# net.cuda()



# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]
# crop_size = 224  
# train_transform = transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.RandomSizedCrop(crop_size), transforms.ToTensor(),
#      transforms.Normalize(mean, std)])
# test_transform = transforms.Compose(
#     [transforms.CenterCrop(crop_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

# train_data = Scene_Classification(DATA_ROOT, train='train', transform=train_transform, useSSD=True, resize=256)
# val_data = Scene_Classification(DATA_ROOT, train='val', transform=test_transform, useSSD=True, resize=256)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=24, shuffle=True,
#                                            num_workers=4, pin_memory=True)
# val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False,
#                                           num_workers=4, pin_memory=True)


# # optimizer = torch.optim.SGD([{'params': net.features.parameters(), 'lr': base_lr},
# #     {'params': net.classifier.parameters(), 'lr': base_lr}, {'params':net.fc1.parameters(), 'lr':base_lr}], momentum=0.9,
# #                             weight_decay=0.0005, nesterov=True)
# optimizer = torch.optim.SGD([{'params': net.features.parameters(), 'lr': base_lr},
#     {'params': net.classifier.parameters(), 'lr': base_lr}], momentum=0.5,
#                             weight_decay=0.0005, nesterov=True)

# # optimizer_adam = torch.optim.Adam([{'params': net.features.parameters(), 'lr': base_lr},
# #     {'params': net.classifier.parameters(), 'lr': base_lr}, {'params':net.fc1.parameters(), 'lr':base_lr}])
# state = {}
# state['epoch'] = 0
# state['test_accuracy'] = 0.0
# state['test_accuracy_t3'] = 0.0


# net.train()
# loss_avg = 0.0
# for batch_idx, (data, target, otherfea) in enumerate(train_loader):
#     data = torch.autograd.Variable(data.cuda())
#     target = torch.autograd.Variable(target.cuda())
#     otherfea = torch.autograd.Variable(otherfea.cuda())

#     # target = target.view(target.size()[0],1)

#     # forward
#     output = net(data)
#     # output = net(data, otherfea)

#     #backward
#     optimizer.zero_grad()


#     turncated_loss_module = TruncatedLoss()
#     loss = turncated_loss_module(output, target)

#     loss.backward()
#     optimizer.step()
#     # optimizer_adam.step()


#     # calculate the loss moving average
#     loss_avg = loss_avg * 0.2 + loss.data[0] * 0.8
#     if batch_idx % display == 0:
#         toprint = 'epoch: {}, batch id: {}, loss: {}, avg_loss: {}, accuracy: {}, acccyracy_t3: {}'.format(state['epoch'], batch_idx, round(loss.data[0], 5), round(loss_avg, 5), round(state['test_accuracy'],3), round(state['test_accuracy_t3'],3))
#         print(toprint)
#         with open(log, 'a') as f:
#             f.write(toprint+'\n')
#     state['train_loss'] = loss_avg
#     break



# y = one_hot(target, output.size(1))

# logit = F.softmax(output)
# logit1 = truncate(logit, 0.8)
# loss = -1 * y * torch.log(logit1) # cross entropy
# loss1 = loss.sum()/float(output.size(0))