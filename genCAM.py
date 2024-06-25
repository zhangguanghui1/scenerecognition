from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
import torch
from matplotlib import pyplot as plt
import numpy as np


from torchvision.models import resnet50
from RGBBranch import RGBBranch
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


from clip_vit import VitResNet,VisionTransformer


def ReshapeTransform(tensor, height=14, width=14):
    # 去掉类别标记
    print("shape",tensor.shape)#shape torch.Size([197, 1, 768])

    result = tensor[1:, :, :].reshape(-1,height, width, tensor.size(2))
    #result = tensor[:, 1:, :].reshape(-1,height, width, tensor.size(2))
    print("result",result.shape)
    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens,size))
    if titles == False:
        titles="0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def tensor2img(tensor,heatmap=False,shape=(224,224)):
    np_arr=tensor.detach().numpy()#[0]
    #对数据进行归一化
    if np_arr.max()>1 or np_arr.min()<0:
        np_arr=np_arr-np_arr.min()
        np_arr=np_arr/np_arr.max()
    #np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0]==1:
        np_arr=np.concatenate([np_arr,np_arr,np_arr],axis=0)
    np_arr=np_arr.transpose((1,2,0))
    return np_arr
 
# path=r"/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397/train/abbey/sun_aaimforvxlklilzm.jpg"

# bin_data=torchvision.io.read_file(path)#加载二进制数据
# img=torchvision.io.decode_image(bin_data)/255#解码成CHW的图片
# img=img.unsqueeze(0)#变成BCHW的数据，B==1; squeeze
# input_tensor=torchvision.transforms.functional.resize(img,[224, 224])
 
# #对图像进行水平翻转，得到两个数据
# input_tensors=torch.cat([input_tensor, input_tensor.flip(dims=(3,))],axis=0)
#反向梯度传播是从最后预测开始，经过整个模型。target_layers只是表示记录这些layers的梯度信息而已
data_transform = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


img_path = "/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Datasets/SUN397/train/airport_terminal/sun_aaenwecffknbumjq.jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

img = Image.open(img_path).convert('RGB')
#img = np.array(img, dtype=np.uint8)
# [C, H, W]
img_tensor = data_transform(img)

# expand batch dimension
# [C, H, W] -> [N, C, H, W]
input_tensors = torch.unsqueeze(img_tensor, dim=0)
print("input shape",input_tensors.shape)

print("cam")

#model =VitVersion(scene_classes=397)
# for key in model.state_dict().keys():
#     print(key)

model =VitResNet()
model.load_state_dict(torch.load('vitresnet_model/res18_vit_attmod freeze 43 epoch-79.466.pth'))
# dict=torch.load('/root/data/cfm/sceneReg/CLIP-main/vitversion/vit-d_c_auto-res_79.637.pth')
# for key in dict.keys():
#     print(key)
# model = RGBBranch("ResNet-18", 397)
# checkpoint = torch.load('/root/data/cfm/sceneReg/Semantic-Aware-Scene-Recognition/Data/Model_Zoo/SUN397/RGB_ResNet18_SUN.pth.tar')
# model.load_state_dict(checkpoint['state_dict'])


print("model",model)
#model = resnet50(pretrained=True)
#target_layers = [model.encoder4[-1]]# 要查看ResNet分支
#target_layers = [model.lastConvRGB2[-1]]

target_layers = [model.vit.transformer.resblocks[-2].ln_1]#设置要查看的层为vit分支

#---vit最后只对class_token做预测，只用它对结果有贡献，也就只有它有梯度，再将最后预测的结果进行反向传播，后面那几层都只是token自己的MLP,LN只有在多头注意力才将class_token与其余token关联起来
#反向梯度传播是从最后预测开始，经过整个模型。target_layers只是表示记录这些layers的梯度信息而已

#cam = GradCAM(model=model,target_layers=target_layers,use_cuda=False)#查看resnet分支用这个

cam = GradCAM(model=model,target_layers=target_layers,use_cuda=False,reshape_transform=ReshapeTransform)#查看vit分支用这个

grayscale_cams = cam(input_tensor=input_tensors, targets=None)#targets=None 自动调用概率最大的类别显示
for grayscale_cam,tensor in zip(grayscale_cams,input_tensors):
    #将热力图结果与原图进行融合
    rgb_img=tensor2img(tensor)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    myimshows([rgb_img, grayscale_cam, visualization],["image","cam","image + cam"])

