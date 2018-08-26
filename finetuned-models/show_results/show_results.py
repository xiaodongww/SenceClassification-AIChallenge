import cPickle as pickle
import numpy as np
GROUNDTRUTH_FILE = 'val.txt'
groundtruth = []
with open(GROUNDTRUTH_FILE) as f:
    for line in f:
        groundtruth.append(int(line.split(' ')[1].strip()))



def getAcc(pred_probs, groundtruth):
    sample_num = len(groundtruth)
    correct = 0
    correct_t3 = 0

    for i  in range(sample_num):
        if pred_probs[i].argmax() == groundtruth[i]:
            correct += 1

        if groundtruth[i] in pred_probs[i].argsort()[-3:][::-1]:
            correct_t3 += 1
    return correct/float(sample_num), correct_t3/float(sample_num)


new1 = 'Places2-365-CNN-deploy1202-val-10crop.pickle'
new2 = 'Places2-401-CNN-deploy1202-val-10crop.pickle'
new3 = 'Places205-VGGNet-19-deploy1202-val-10crop.pickle'
new4 = 'SE_RESNET_50-step2-deploy1202-val-10crop.pickle'
new5 = 'ResNet152-places365-deploy1203-val-10crop.pickle'
new6 = 'densenet161_ckpt_epoch_48_tencrop_softmax_val_wxd.pickle'
new7 = 'atdensenet161_ckpt_epoch_26_tencrop_softmax_val_wxd.pickle'

new = [new1, new2, new3, new4, new5, new6, new7]
new_probs = []
for each in new:
    with open(each) as f:
        new_probs.append(pickle.load(f))

for i in range(len(new_probs)):
    acc, acc_t3 = getAcc(new_probs[i], groundtruth)   
    print(new[i],acc_t3)   

