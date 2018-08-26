import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from scene_classification import Scene_Classification

def one_hot(target, classes):
    mask = torch.rand(target.size(0),classes).zero_()
    mask.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
    mask = torch.autograd.Variable(mask.cuda())

    return mask


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(1))

        logit = F.softmax(input)
        # logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * ((1 - logit) ** self.gamma) / float(input.size(0))# focal loss

        return loss.sum()

#  the folling lines is used for debugging
# from models import densenet
# import torchvision.transforms as transforms
# DATA_ROOT = '/home/wuxiaodong/ai_challenge/data/'
# state_dict = torch.load('./snapshots/ckpt_epoch_48.pytorch')
# net = densenet.densenet161(state_dict)
# net.cuda()
# mean = [x / 255 for x in [125.3, 123.0, 113.9]]
# std = [x / 255 for x in [63.0, 62.1, 66.7]]
# crop_size = 224
# train_transform = transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.RandomSizedCrop(crop_size), transforms.ToTensor(),
#      transforms.Normalize(mean, std)])
# train_data = Scene_Classification(DATA_ROOT, train='train', transform=train_transform, useSSD=True, resize=256)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True,
#                                            num_workers=4, pin_memory=True)

# data, target, otherfea = next(iter(train_loader))
# data = torch.autograd.Variable(data.cuda())
# target = torch.autograd.Variable(target.cuda())
# otherfea = torch.autograd.Variable(otherfea.cuda())
# output = net(data)
# mask = one_hot(target, output.size(1))


# loss_cross = F.cross_entropy(output, target)

# logit = F.softmax(output)
# loss = -1 * mask * torch.log(logit)
# gamma = 1
# loss = loss * ((1 - logit) ** gamma)/float(output.size(0))