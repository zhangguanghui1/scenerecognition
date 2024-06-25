import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

import torch
import numpy as np
from torch import nn
from pathlib import Path 
import torch.nn.functional as F
import math
from torch.utils.tensorboard import SummaryWriter

from SUN397Dataset import SUN397Dataset
#from Places365Dataset import Places365Dataset

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, \
    RandomResizedCrop
from PIL import Image

from torch import optim
import torch.backends.cudnn as cudnn

from clip_vit import VitResNet, VisionTransformer
from RGBBranch import RGBBranch

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def train(train_loader,val_loader, model, optimizer, train_epoch, traintype="freeze"):
    
    cross_entropy = nn.CrossEntropyLoss()
    #val_acc_list = []
    # out_dir = "vitres_dataimprove/" 
    out_dir = "vitres_result_zsgdunfreeze/" 
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

        if acc_val > best: 
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

def test(model,val_loader,tencrop=False):
    all_predicts=[]
    all_labels=[]
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch_idx, data in enumerate(val_loader):
            model.eval()
            images = data['Image'].cuda()
            labels = data['Scene Index'].cuda()

            #tencrop
            if tencrop:
                bs, ncrops, c_img, h, w = images.size()
                images = images.view(-1, c_img, h, w)

            outputs= model(images)

            if tencrop:
                outputs = outputs.view(bs, ncrops, -1).mean(1)

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
        #autoaugment.RandAugment(interpolation=BICUBIC),
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
    traindir = "/root/data/zgh/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397" 
    valdir = "/root/data/zgh/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397" 

    train_dataset = SUN397Dataset(traindir, "train",train_transform())
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=False) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=False) 

    val_dataset = SUN397Dataset(valdir, "val", test_transform(),tencrops=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=0, pin_memory=False) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=2, pin_memory=False) 

  
    # train_dataset = Places365Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/places365_standard", "train",train_transform())
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True, num_workers=4, pin_memory=False)

    # val_dataset = Places365Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/places365_standard", "val", test_transform(),tencrops=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=4, pin_memory=False)
    
    
    # train_dataset = MITIndoor67Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/MITIndoor67", "train",train_transform())
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,shuffle=True, num_workers=2, pin_memory=False)

    # val_dataset = MITIndoor67Dataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/MITIndoor67", "val", test_transform(),tencrops=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=2, pin_memory=False)
    
    # train_dataset = ADE20KDataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/ADEChallengeData2016", "training",train_transform())
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100,shuffle=True, num_workers=4, pin_memory=False)

    # val_dataset = ADE20KDataset("/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/ADEChallengeData2016", "validation", test_transform(),tencrops=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,num_workers=4, pin_memory=False)
 

    print("vitresnet") 
    model = VitResNet()  
    # model.load_state_dict(torch.load('/root/data/zgh/sceneReg/vit-scene/vitresnet_model/res18_vit_attmod freeze 43 epoch-79.466.pth'))

    model_dict = model.state_dict() 
    # #
    # # # for key in model_dict.keys():
    # # #     print(key)
    # #
    premol = RGBBranch("ResNet-18", 397)
    checkpoint = torch.load('/root/data/zgh/sceneReg/Semantic-Aware-Scene-Recognition/Data/Model_Zoo/SUN397/RGB_ResNet18_SUN.pth.tar')
    premol.load_state_dict(checkpoint['state_dict'])
    predict = premol.state_dict()
    #
    #
    #
    for k, v in predict.items():
        flag = False
        for ss in model_dict.keys():
            if k in ss and k not in ['fc.weight','fc.bias']:
                # print(" k ",k)
                # print(" ss ",ss)
                s = ss
                flag = True
                break
            else:
                continue
        if flag:
            #print("freeze")
            model_dict[s] = predict[k]
            # print("predict[k].requires_grad:",predict[k].requires_grad)
            # print("model_dict[s].requires_grad:",model_dict[s].requires_grad)
            # print("s:",s)
            model_dict[s].requires_grad = False # 这是设置为冻结了. 
            #print("af model_dict[s].requires_grad",model_dict[s].requires_grad)
    
    #
    vitmodel = VisionTransformer() 
    # vit_dict = vitmodel.state_dict() 
    vitstate_dict = torch.load('/root/data/zgh/sceneReg/vit-scene/previt.pth')
    vitmodel.load_state_dict(vitstate_dict['model'])
    previt = vitmodel.state_dict()
    
    for k, v in previt.items():
        flag = False
        for ss in model_dict.keys():
            if k in ss and 'vit' in ss:
                # print(" k ",k)
                # print(" ss ",ss)
                s = ss
                flag = True
                break
    
            else:
                continue
        if flag:
            #print("freeze")
            model_dict[s] = previt[k]
            model_dict[s].requires_grad = False
    
    model.load_state_dict(model_dict) 
    

    #optimizer = optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=2.5e-4, momentum=0.9, weight_decay=5E-5)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20*len(train_loader))
    
    freeze_epoch = 300
    unfreeze_epoch = 50

    if USE_CUDA:
        # model = torch.nn.DataParallel(model.cuda(), device_ids=[0,1]).cuda()
        model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = USE_CUDA
    
    
    # #### ------------- freeze train ---------------- #### 
    # print("freeze train:") 
    # model.module.freeze()

    # # for name, param in model.named_parameters():
    # #         if param.requires_grad:
    # #             print("requires_grad: True ", name)
    # #         else:
    # #             print("requires_grad: False ", name)
    
    
    # # optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001, eps=1e-4)
    # optimizer = optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=2.5e-4, momentum=0.9, weight_decay=5E-5) 

    # train(train_loader, val_loader, model, optimizer, freeze_epoch, traintype="freeze") 


    ### ------------- unfreeze train ---------------- #### 
    print("unfreeze train:") 
    model.module.unfreeze() 

    # optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001, eps=1e-4)
    optimizer = optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=2.5e-4, momentum=0.9, weight_decay=5E-5)

    train(train_loader, val_loader, model, optimizer, unfreeze_epoch, traintype="unfreeze") 

if __name__ == '__main__':
    
    main()





