from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        # ----------------------#
        # 多头注意力机制
        # ----------------------#
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        # -------------------------------------------------------------------#
        # 在MLP层中首先是进行一次全连接,之后是过QuickGELU激活函数,最后是通过投影进行映射
        # -------------------------------------------------------------------#

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    # -------------------------------------#
    # 该函数的作用是对输入的张量使用多头注意力机制
    # -------------------------------------#
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    # ---------------------------------------------------------------#
    # 在这个前向传播函数中,对于transformer模块进行了定义以及说明
    # ---------------------------------------------------------------#
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class VisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512,num_classes=397):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # print("input_resolution :",input_resolution)#224
        # print("patch_size :", patch_size)#16
        # print("width :", width)#768
        # print("layers: :", layers)#12
        # print("heads :", heads)#12
        # print("output_dim :", output_dim)#512


        scale = width ** -0.5  #** 代表乘方
        # ------------------------------------------#
        # 在这里我们可以用nn.Parameter()来将这
        # 个随机初始化的Tensor注册为可学习的参数Parameter
        # ------------------------------------------#
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        #self.head = nn.Linear(output_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # -----------------------------------------------------------------------------------------#
        # 此处的卷积可以将张量的shape转变为batch_size,width,grid,grid(grid=input_resolution/patch_size)
        # -----------------------------------------------------------------------------------------#
        # x=self.start(x)
        # x=self.cbma(x)

        x = self.conv1(x)  # shape = [*, width, grid, grid] torch.Size([100, 768, 14, 14])

        #print('x.shape 1:',x.shape)[100, 768, 14, 14]

        # ---------------------------------------------#
        # reshape之后,shape=batch_size,width,grid ** 2
        # ---------------------------------------------#
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # ----------------------------#
        # 转置之后,shape为 batch_size,grid ** 2,width
        # ----------------------------#
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # ------------------------------------------------------#
        # pass这条语句之后,shape=batch_size,grid ** 2 + 1,width
        # ------------------------------------------------------#
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # --------------------------------------------#
        # 加上其位置编码信息,并且pass through LayerNorm层
        # --------------------------------------------#
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # -----------------------------------------------#
        # shape先转变为grid ** 2 + 1,batch_size,width
        # 之后经由transformer结构编码
        # 最后再进行转置,恢复为batch_size,grid ** 2 + 1,width
        # 再pass through LayerNorm层
        # -----------------------------------------------#
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        # -------------------------#
        # 若成立则将会进行矩阵乘法运算
        # -------------------------#

        if self.proj is not None:
            x = x @ self.proj

        #x = self.head(x)
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out,ksize=3):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=ksize,groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class VitResNet(nn.Module):
    def __init__(self, scene_classes=397):
        super(VitResNet, self).__init__()

        base = resnet.resnet18(pretrained=True)
        
        sizes_lastConv = [512, 512, 512]
        
        # --------------------------------#
        #           RGB Branch            #
        # ------------------------------- #
        # First initial block
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        )
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # --------------------------------#
        #          vit Branch          #
        # ------------------------------- #
        self.vit=VisionTransformer()

        # RGB Scene Classification Layers.
        self.fc_RGB = nn.Linear(512, scene_classes)

        # --------------------------------#
        #         Attention Module        #
        # ------------------------------- #
        # Final Scene Classification Layers
        # self.lastConvRGB1 = nn.Sequential(
        #     nn.Conv2d(sizes_lastConv[0], sizes_lastConv[1], kernel_size=3, bias=False),
        #     nn.BatchNorm2d(sizes_lastConv[1]),
        #     nn.ReLU(inplace=True),
        # )
        self.lastConvRGB2 = nn.Sequential(
            depthwise_separable_conv(512,1024,5),
            #nn.Conv2d(sizes_lastConv[2], 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        # self.lastConvVIT1 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # )
        self.lastConvVIT2 = nn.Sequential(
            depthwise_separable_conv(512,1024,5),
            #nn.Conv2d(512, 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # --------------------------------#
        #            Classifier           #
        # ------------------------------- #
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.avgpool7 = nn.AvgPool2d(7, stride=1)
        self.avgpool3 = nn.AvgPool2d(3, stride=1)

        self.fc = nn.Linear(1024, scene_classes)

    def forward(self, x):
        
        # --------------------------------#
        #           RGB Branch            #
        # ------------------------------- #
        x1, pool_indices = self.in_block(x)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        #print(" e4 shape:",e4.shape)#[16,512,7,7]

        # RGB Classification Layer
        act_rgb = self.avgpool7(e4)
        act_rgb = act_rgb.view(act_rgb.size(0), -1)
        act_rgb = self.dropout(act_rgb)
        act_rgb = self.fc_RGB(act_rgb)

        # --------------------------------#
        #        vit Branch          #
        # ------------------------------- #
        y=self.vit(x)#[16,512]

        y=y.unsqueeze(2)
        y=y.unsqueeze(2)
        y = F.interpolate(y, scale_factor=7, mode='bilinear', align_corners=True)#w h

        # --------------------------------#
        #         Attention Module        #
        # ------------------------------- #
        #print(" e4 shape:",e4.shape)#[16,512,7,7]
        #e5 = self.lastConvRGB1(e4)
        e6 = self.lastConvRGB2(e4)
        #print(" e6 shape:",e6.shape)#[32, 1024, 5, 5]

        #print(" y shape:",y.shape)#[16,512,7,7]
        #y1 = self.lastConvVIT1(y)
        y2 = self.lastConvVIT2(y)

        #print(" y2 shape:",y2.shape)#[32, 1024, 5, 5]
        # Attention Mechanism
        e7 = y2 * self.sigmoid(e6)



        # --------------------------------#
        #            Classifier           #
        # ------------------------------- #
        e8 = self.avgpool3(e7)
        act = e8.view(e8.size(0), -1)
        act = self.dropout(act)#

        act = self.fc(act)

        return act

    def loss(self, x, target):
        # Check inputs
        assert (x.shape[0] == target.shape[0])

        # Classification loss
        loss = self.criterion(x, target.long())

        return loss

    def freeze(self):
        for name, param in self.named_parameters():
            #print("name:",name)
            if "in_block" in name:
                param.requires_grad = False

            if "encoder" in name:
                param.requires_grad = False

            if "vit" in name:
                param.requires_grad = False
        
        # 查看是否关闭成功
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print("requires_grad: True ", name)
        #     else:
        #         print("requires_grad: False ", name)


    def unfreeze(self):
        for name, param in self.named_parameters():
            #print("name:",name)
            if "in_block" in name:
                param.requires_grad = True

            if "encoder" in name:
                param.requires_grad = True

            if "vit" in name:
                param.requires_grad = True
        
        # 查看是否关闭成功
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print("requires_grad: True ", name)
        #     else:
        #         print("requires_grad: False ", name)



