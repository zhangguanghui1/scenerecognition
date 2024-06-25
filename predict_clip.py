import sys
sys.path.append('/root/data/cfm/sceneReg/CLIP-main')
import numpy as np

import os
# import clip
import torch
from SUN397Dataset import SUN397Dataset
import torch.utils.data
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from torch.utils.data import DataLoader
import joblib
from sklearn.svm import SVC
from PIL import Image
import time
from vit_version import VitVersion,VisionTransformer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# from clip_vit import VitResNet,VisionTransformer
def test_transform():
    #print("_transform ----")
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_classes(file):
    classes=[]
    class_file_name = os.path.join(file, "scene_names.txt")
    with open(class_file_name) as class_file:
        for line in class_file:# line: /a/abbey 0   /c/casino-indoor 81
            line = line.split()[0]
            split_indices = [i for i, letter in enumerate(line) if letter == '/'] #记录为'/'的下标
            # Check if there a class with a subclass inside (outdoor, indoor)
            if len(split_indices) > 2:
                line = line[:split_indices[2]] + '-' + line[split_indices[2]+1:]  
            classes.append(line[split_indices[1] + 1:])
    return classes

def read_directory(directory_name,preprocess):
    img_list=[]
    for filename in os.listdir(directory_name):
        #print(filename)  # 仅仅是为了测试
        img = Image.open(directory_name + "/" + filename)
        img=preprocess(img)
        #print("img size:",img.size())
        img_list.append(img)
    return img_list


classes=get_classes("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397")
#print("classes:",classes)
print("Download the image")


# img = Image.open("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397/val/abbey/sun_aaimforvxlklilzm.jpg")
# img=preprocess(img)
# img=img.unsqueeze(0)


img_list=read_directory("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397/val/abbey",test_transform())


# print("onload svm")
# classifier = joblib.load('lr.model')
print("onload vit model")
s1 = time.time()

# model =VitResNet()
# model.load_state_dict(torch.load('/root/data/cfm/sceneReg/CLIP-main/vitresnet_model/res18_vit_attmod freeze 43 epoch-79.466.pth'))


model =VitVersion()#
model.load_state_dict(torch.load('/root/data/cfm/sceneReg/CLIP-main/vitversion/7_d_c_freeze_99_79.239.pth'))


s2 = time.time()
print("load time:",s2-s1)

start = time.time()
#print("img size:",img.size())
#print("Calculate the image features")
#all_features = []

with torch.no_grad():
    for i in range(len(img_list)):

        img=img_list[i].unsqueeze(0)
        # features = model.encode_image(img)
        # all_features.append(features)
        outputs= model(img) 
        _, predicted = torch.max(outputs.data, dim=1)
        #print("predict:",predicted)
        print("predict:",classes[predicted[0]])


# for i in range(len(all_features)):

#     predict = classifier.predict(all_features[i])
#     print("predict:",classes[predict[0]])


end = time.time()
# print("len(all_features):",len(all_features))
print("time:",end-start)
print("time:",(end-start)/len(img_list))
