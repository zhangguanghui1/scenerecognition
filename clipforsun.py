
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import clip
import torch
from torchvision.datasets import CIFAR100
# from SUN397Dataset import SUN397Dataset
from Places365Dataset import Places365Dataset
#from ADE20KDataset import ADE20KDataset
# from MITIndoor67Dataset import MITIndoor67Dataset
from PIL import Image

import torch.utils.data
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from torch.utils.data import DataLoader
import joblib
from sklearn.svm import SVC
from clip_vit import VisionTransformer 
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, \
    RandomResizedCrop,autoaugment

def train_transform():
    #print("_transform ----")
    return Compose([
        RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        RandomHorizontalFlip(),
        autoaugment.RandAugment(interpolation=BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def test_transform():
    #print("_transform ----")
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
premodel, preprocess = clip.load('ViT-B/16', device)

print(" VisionTransformer ")
model= VisionTransformer()
model_dict = model.state_dict()
predict = premodel.visual.state_dict()

for k, v in predict.items():
    flag = False
    for ss in model_dict.keys():
        if k in ss:
            #print(" k ",k)
            s = ss
            flag = True
            break
            
        else:
            continue
    if flag:
        model_dict[s] = predict[k] 

model.load_state_dict(model_dict)
model.to(device)


#torch.save({'model': model.state_dict()}, 'previt.pth')

# ## 读取模型
state_dict = torch.load('previt.pth')
model.load_state_dict(state_dict['model'])


# print("Download the dataset")
# traindir = "/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397"
# valdir = "/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397"

# train_dataset = SUN397Dataset(traindir, "train",preprocess)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,shuffle=True, num_workers=0, pin_memory=False)

# val_dataset = SUN397Dataset(valdir, "val", preprocess,tencrops=False)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=0, pin_memory=False)

train_dataset = Places365Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/places365_standard", "train",train_transform())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True, num_workers=1, pin_memory=False)

val_dataset = Places365Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/places365_standard", "val", test_transform())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=1, pin_memory=False)
    
# train_dataset = MITIndoor67Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/MITIndoor67", "train",train_transform())
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,shuffle=True, num_workers=4, pin_memory=False)

# val_dataset = MITIndoor67Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/MITIndoor67", "val", test_transform())
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=4, pin_memory=False)
    
# train_dataset = ADE20KDataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/ADEChallengeData2016", "training",train_transform())
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,shuffle=True, num_workers=4, pin_memory=False)

# val_dataset = ADE20KDataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/ADEChallengeData2016", "validation", test_transform())
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=4, pin_memory=False)
    
    



classes = train_dataset.classes


def get_features(model,dataset):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataset):
            RGB_image = mini_batch['Image']
            sceneLabelGT = mini_batch['Scene Index']

            # features = model.encode_image(RGB_image.to(device))
            features = model(RGB_image.to(device))
            #print("features shape",features.shape)#[100, 512]
            all_features.append(features)

            all_labels.append(sceneLabelGT)


    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def topkaccuracy(output, target, topk=(1,2,5)):
    """
    Computes the top-k accuracy between output and target.
    :param output: output vector from the network
    :param target: ground-truth
    :param topk: Top-k results desired, i.e. top1, top5, top10
    :return: vector with accuracy values
    """
    maxk = max(topk)
    batch_size = len(target)
    
    output = torch.from_numpy(output)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()#转置
    
    target = torch.from_numpy(target)

    print("target:", target.expand_as(pred).size())##[5,19850]
    correct = pred.eq(target.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def getclassAccuracy(output, target, nclasses=397, topk=(1,2,5)):
    maxk = max(topk)
    output = torch.from_numpy(output)
    print("output size:",output.size())
    score, label_index = output.topk(maxk, 1, largest=True, sorted=True)
    
    target = torch.from_numpy(target)
    print("target size:",target.size())
    correct = label_index.eq(torch.unsqueeze(target, 1))

    ClassAccuracyRes = []
    for k in topk:
        ClassAccuracy = torch.zeros([1, nclasses], dtype=torch.uint8).cuda()
        correct_k = correct[:, :k].sum(1)
        for n in range(target.shape[0]):
            ClassAccuracy[0, target[n]] += correct_k[n].byte()
        ClassAccuracyRes.append(ClassAccuracy)

    return ClassAccuracyRes

print("PLACES")

print("Calculate the image features")
train_features, train_labels = get_features(model,train_loader)
test_features, test_labels = get_features(model,val_loader)


print("Perform logistic regression")
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# print("download svm")
# classifier = SVC(max_iter=1000)#,probability=True
# classifier.fit(train_features, train_labels)
#joblib.dump(classifier, 'lr.model')
# lr = joblib.load('svm.model')
# pre =lr.predict_proba(test_features)

# print("Evaluate using the logistic regression classifier")
# predictions = classifier.predict_proba(test_features)
#print("pred: ",predictions)


pre = classifier.predict(test_features)
accuracy = np.mean((test_labels == pre).astype(np.float)) * 100.
print(f"Accuracy = {accuracy:.3f}")

#print("test label ",test_labels)
predictions= classifier.predict_proba(test_features)
topk = topkaccuracy(predictions,test_labels)
print("topk: ",topk)

# classtopk = getclassAccuracy(pre,test_labels)

# ClassTPs_Top1 = classtopk[0].cpu().numpy()
# ClassTPs_Top2 = classtopk[1].cpu().numpy()
# ClassTPs_Top5 = classtopk[2].cpu().numpy()
# # Save Validation Class Accuracy
# val_ClassAcc_top1 = (ClassTPs_Top1 / (50+ 0.0001)) * 100
# np.savetxt('./ValidationTop1ClassAccuracy.txt', np.transpose(val_ClassAcc_top1), '%f')

# val_ClassAcc_top2 = (ClassTPs_Top2 / (50 + 0.0001)) * 100
# np.savetxt('./ValidationTop2ClassAccuracy.txt', np.transpose(val_ClassAcc_top2), '%f')

# val_ClassAcc_top5 = (ClassTPs_Top5 / (50 + 0.0001)) * 100
# np.savetxt('./ValidationTop5ClassAccuracy.txt', np.transpose(val_ClassAcc_top5), '%f')

