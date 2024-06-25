一、数据集准备
按照论文(https://github.com/vpulab/Semantic-Aware-Scene-Recognition)要求下载4个场景识别数据集SUN397（重点数据集，可从百度网盘下载：XXTODO）、MITIndoor67、ADE20K、Places365 
SUN397Dataset.py、MITIndoor67Dataset.py、ADE20KDataset.py、Places365Dataset.py 是加载数据集的对应文件

环境见README-update-.md. 重要的：ubuntu18.04 cuda11.3 pytorch 1.12.0+cu113

二、推理测试
1. VIT+ResNet版本
   python test_vitresnet_z.py 
   加载的模型为/root/data/zgh/sceneReg/vit-scene/vitresnet_model/res18_vit_attmod freeze 43 epoch-79.466.pth 
   output:
    start val:
    Val's ac is: 79.466%
    topk [tensor(79.4660), tensor(90.4635), tensor(97.0932)] 
    
   加载的模型为/root/data/zgh/sceneReg/vit-scene/vitres_result_zsgd_valprevitpth/freeze_217_79.698.pth
   output:
    start val:
    Val's ac is: 79.698%
    topk [tensor(79.6977), tensor(90.6801), tensor(97.0277)] 

2. VIT版本
   python test_vitversion_z.py 
   加载的模型为/root/data/zgh/sceneReg/vit-scene/vitversion/7_d_c_freeze_99_79.239.pth 
   output:
    start val:
    Val's ac is: 79.083%
    topk [tensor(79.0831), tensor(90.3678), tensor(96.9572)] 

三、计算耗时
   python infer_time_z.py 
   支持切换VIT+ResNet、VIT版本. 
   VIT+ResNet：0.2s左右 
   VIT: 0.16s左右
    

三、可视化 
1. VIT+ResNet版本 
   python genCAM_z.py 
   查看VIT+ResNet版本的VIT和ResNet可视化结果，支持切换VIT和ResNet分支. 


四、训练过程
1. VIT+ResNet版本
1）准备RGB预训练版本：从https://github.com/vpulab/Semantic-Aware-Scene-Recognition 下载的在SUN397上RGB分支的模型RGB Branch；
2）准备ViT预训练版本：python /root/data/zgh/sceneReg/vit-scene/gen_previtpth.py 
3）python train_vitresnet_zsgd.py 训练时长约24h. 

附：
clipforsun.py
1.用于测试clip模型中vit作为特征提取器，LogisticRegression作为分类器的场景识别的检测效果，previt.pth保存clip模型中的vit模型，lr.model保存对应的LogisticRegression模型
2.用于测试clip模型中vit作为特征提取器，SVM作为分类器的场景识别的检测效果，svm.model保存的对应模型

感谢：
https://github.com/vpulab/Semantic-Aware-Scene-Recognition
https://github.com/openai/CLIP
