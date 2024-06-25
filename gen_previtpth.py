
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import clip
import torch
from PIL import Image
from clip_vit import VisionTransformer 
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC 

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

# torch.save({'model': model.state_dict()}, 'previt.pth') 
torch.save({'model': model.state_dict()}, '/root/data/zgh/sceneReg/vit-scene/previt_tmp.pth') 