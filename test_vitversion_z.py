import torch
import numpy as np
from torch import nn
from pathlib import Path
import os
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter

from SUN397Dataset import SUN397Dataset
#from Places365Dataset import Places365Dataset
#from ADE20KDataset import ADE20KDataset
#from MITIndoor67Dataset import MITIndoor67Dataset

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, \
    RandomResizedCrop,autoaugment
from PIL import Image

from torch import optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from vit_version import VitVersion, VisionTransformer

from RGBBranch import RGBBranch
# from transmix import Mixup_transmix
# from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

os.environ["CUDA_VISIBLE_DEVICES"]="0,1" 

# mixup_fn = Mixup_transmix() 

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


#使用mixup时的数据集融合器，输入数据集的输入（一批图片），以及对应标签，返回线性相加后的图片
def mixup_data(x, y, alpha=1.0):
    #随机生成一个 beta 分布的参数 lam，用于生成随机的线性组合，以实现 mixup 数据扩充。
    lam = np.random.beta(alpha, alpha)
    #生成一个随机的序列，用于将输入数据进行 shuffle。
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    #得到混合后的新图片
    mixed_x = lam * x + (1 - lam) * x[index, :]
    #得到混图对应的两类标签
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train(train_loader,val_loader, model, optimizer,train_epoch,traintype="freeze"):
    
    cross_entropy = nn.CrossEntropyLoss()
    #val_acc_list = []
    out_dir = "vitversion/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    best = 0
    for epoch in range(0, train_epoch):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()

        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            length = len(train_loader)
            images = data['Image'].cuda()
            labels = data['Scene Index'].cuda()
            
            #mixup扩充数据集
            # images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)
                        
            # outputs = model(images)
            
            # # #这里对应调整了使用mixup后的数据集的loss
            # loss = lam * cross_entropy(outputs, targets_a) + (1 - lam) * cross_entropy(outputs, targets_b)
            
            outputs= model(images) 
            loss = cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            #print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))
        
        print('[epoch:%d] Loss: %.03f | Acc: %.3f%% ' 
                % (epoch + 1, sum_loss / (batch_idx + 1), 100. * correct / total))
        


        #get the ac with testdataset in each epoch
        print('Waiting Val...')
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for batch_idx, data in enumerate(val_loader):
                model.eval()
                images = data['Image'].cuda()
                labels = data['Scene Index'].cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('Val\'s ac is: %.3f%%' % (100 * correct / total))
            
            acc_val = 100 * correct / total
            #val_acc_list.append(acc_val) 

        if acc_val>best:
            best=acc_val
            torch.save(model.module.state_dict(), out_dir+traintype+'_%d_%.3f.pth' % (epoch,acc_val))

        # if acc_val == max(val_acc_list):
        #     torch.save(model.state_dict(), out_dir+"best.pt") 
        #     print("save epoch {} model".format(epoch)) 


def topkaccuracy(output, target, topk=(1,2,5)):
    maxk = max(topk)
    batch_size = len(target)
    
    output = torch.from_numpy(output)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t() # 转置
    
    target = torch.from_numpy(target)

    #print("target:", target.expand_as(pred).size())##[5,19850]
    correct = pred.eq(target.expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test(model,val_loader):
    all_predicts=[]
    all_labels=[]
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch_idx, data in enumerate(val_loader):
            model.eval()
            images = data['Image'].cuda()
            labels = data['Scene Index'].cuda()
            outputs= model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum()

            all_predicts.append(outputs.data)
            all_labels.append(labels)

        print('Val\'s ac is: %.3f%%' % (100 * correct / total))
        
    all_predicts=torch.cat(all_predicts).cpu().numpy()
    all_labels=torch.cat(all_labels).cpu().numpy()
    topk=topkaccuracy(all_predicts,all_labels)
    print("topk",topk)



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


def main():
    
    
    USE_CUDA = torch.cuda.is_available()
    
    print("Download the places dataset") 
    valdir = "/root/data/zgh/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397" 
    
    val_dataset = SUN397Dataset(valdir, "val", test_transform())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=False) 

    # val_dataset = Places365Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/places365_standard", "val", test_transform())
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=4, pin_memory=False)

    # val_dataset = MITIndoor67Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/MITIndoor67", "val", test_transform())
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=4, pin_memory=False)

    # val_dataset = ADE20KDataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/ADEChallengeData2016", "validation", test_transform())
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=4, pin_memory=False)
    
    print("VitVersion")
    model = VitVersion(scene_classes=397) 
    # #### FOR DEBUG #### 
    # model_dict = model.state_dict() 
    # print('key:----------------------------------------') 
    # for key in model_dict.keys():
    #     print(key)
    # #### FOR DEBUG #### 
    
    state_dict=torch.load('/root/data/zgh/sceneReg/vit-scene/vitversion/7_d_c_freeze_99_79.239.pth') 
    # #### FOR DEBUG #### 
    # print('key1:----------------------------------------') 
    # for key1 in state_dict.keys():
    #     print(key1) 
    # #### FOR DEBUG #### 
    model.load_state_dict(state_dict) 
    
    if USE_CUDA:
        model = torch.nn.DataParallel(model.cuda(), device_ids=[0,1]).cuda()
    cudnn.benchmark = USE_CUDA 

    print("start val:") 
    test(model,val_loader) 
    
    
if __name__ == '__main__': 
    
    main() 





