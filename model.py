# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 21:58
# @Author  : Curry
# @File    : model.py

import config
import timm
import torch
import os
os.environ['TORCH_HOME'] = config.PRETRAINED_MODEL_DIR
from torchvision import models
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet

# from model.EfficientNetv2 import efficientnetv2_s


# def build_model(pretrain_flag=False):
#     if pretrain_flag:
#         pretrain_weights = "imagenet"
#     else:
#         pretrain_weights = False
#     model = timm.create_model(config.BACKBONE,
#                               pretrained=pretrain_flag,
#                               num_classes=config.NUM_CLASSES)
#     model = timm.create_model("convnext_large_384_in22ft1k", pretrained=True)
#     model.head.fc = nn.Linear(model.num_features, 2)
#     model.to(config.DEVICE)
#     return model

# 方式二，下载预训练权重到指定目录
# def build_model(pretrain_flag=False):
#     model = timm.create_model(config.BACKBONE,
#                           pretrained=False,
#                           num_classes=config.NUM_CLASSES)
#     model_weights = model.state_dict()
#     pthfile = './pretrained/efficientnet_b3_ra2-cf984f9c.pth'
#     pre_weights = torch.load(pthfile)
#     # delete classifier weights
#     # 这种方法主要是遍历字典，.pth文件（权重文件）的本质就是字典的存储
#     # 通过改变我们载入的权重的键值对，可以和当前的网络进行配对的
#     # 这里举到的例子是对"classifier"结构层的键值对剔除，或者说是不载入该模块的训练权重，这里的"classifier"结构层就是最后一部分分类层
#     pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
#     # 如果修改了载入权重或载入权重的结构和当前模型的结构不完全相同，需要加strict=False，保证能够权重载入
#     model.load_state_dict(pre_dict, strict=False)
#
#     model.to(config.DEVICE)
#     return model

def build_model(pretrain_flag=False):
    # Efficientnetb7
    # model = EfficientNet.from_name(config.BACKBONE)
    # # pthfile = '../pretrained/efficientnet-b7-dcc49843.pth'
    # state_dict = '../pretrained/efficientnet-b7-dcc49843.pth'
    # model.load_state_dict(state_dict)

    # resnext101_64x4d
    if pretrain_flag:
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    else:
        model = models.convnext_base(weights=None)

    model.classifier[2] = nn.Linear(1024, 2)

    # # 修改全连接层
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    model.to(config.DEVICE)
    return model


# def build_model(pretrain_flag=False):
    # model = models.vgg16(pretrained=False)
    # model = models.densenet201(weights=models.DenseNet201_Weights)
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, 2)
    # model = models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    # model = RegNetForImageClassification.from_pretrained("zuppif/regnet-y-040")
    # model = models.vgg19(pretrained=pretrain_flag)
    # model.classifier[6] = nn.Linear(4096, 2)
    # model.to(config.DEVICE)
    # # model.classifier[6] = nn.Linear(4096, 2)
    # # num_ftrs = model.fc.in_features
    # # model.fc = nn.Linear(num_ftrs, 2)
    # model.to(config.DEVICE)
    # return model


