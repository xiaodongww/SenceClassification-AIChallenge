import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import json

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
crop_size = 224
default_transform = transforms.Compose(
    [transforms.CenterCrop(crop_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

def default_loader(path, resize,transform):
    # transform1 = transforms.Compose([transforms.ToTensor()])  
    # transform1 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])  
    img = Image.open(path).convert('RGB')
    img = img.resize((resize,resize))

    return transform(img)



class Scene_Classification(data.Dataset):
    def __init__(self, root, train='train', transform=default_transform, useSSD=False,resize=256):
        self.root = os.path.expanduser(root)

        if train not in ['train', 'val', 'testa', 'testb']:
            raise KeyError

        self.train = train
        self.transform = transform
        print('Loading {} data...'.format(self.train))
        self.imgf_path = os.path.join(self.root, self.train+'_imgs')
        self.names = []
        self.labels = []
        self.ssdfeas = []
        self.resize = int(resize)
        with open(os.path.join(self.root, self.train+'.txt')) as f:
            for line in f:
                name, label = line.strip().split(' ')
                self.names.append(name.strip()) 
                self.labels.append(int(label.strip()))   
        self.length = len(self.names)
        self.useSSD = useSSD
        if useSSD:
            with open(os.path.join(self.root, self.train+'_ssd_fea.txt')) as f:
                ssd_json = json.load(f)
                for key in ssd_json.keys():
                    self.ssdfeas.append(torch.Tensor(ssd_json[key]))
        # self.img =default_loader('2.jpg',resize)


    def __getitem__(self, index):
        img_path = os.path.join(self.imgf_path,self.names[index])
        # print(img_path)
        img = default_loader(img_path,self.resize, self.transform)
        if not self.useSSD:
            return img, self.labels[index]
        else:
            return img, self.labels[index], self.ssdfeas[index]

    def __len__(self):
        return self.length
    
    def getName(self):
        pass

# dset = Scene_Classification('/home/wuxiaodong/ai_challenge/data/','testa')