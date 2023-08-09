# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 21:58
# @Author  : Curry
# @File    : config.py
import random

import torch
import albumentations as A
import cv2
from albumentations.core.transforms_interface import DualTransform
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TAMPERED_IMG_PATH = "../lesson1/icdar_2023_dtt/data/train/tampered"
UNTAMPERED_IMG_PATH = "../lesson1/icdar_2023_dtt/data/train/untampered"
# for Debug
# TAMPERED_IMG_PATH = "../lesson1/icdar_2023_dtt/data/demo/tampered/imgs"
# UNTAMPERED_IMG_PATH = "../lesson1/icdar_2023_dtt/data/demo/untampered"

TEST_IMG_PATH = r"E:\天池\ICDAR文本篡改分类和检测\lesson1\icdar_2023_dtt\data\test\imgs"
# TEST_IMG_PATH = r"E:\天池\ICDAR文本篡改分类和检测\lesson1\icdar_2023_dtt\data\demo\tampered"

# 打标测试集
LABEL_TEST_IMG_PATH = ""

N_FOLD = 5
IMG_SIZE = [512, 512]
# IMG_SIZE = [768, 768]
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = TRAIN_BATCH_SIZE * 2
TEST_BATCH_SIZE = 1
NUM_EPOCH = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
LR_DROP = 8
THREDHOLD = 0.5
NUM_CLASSES = 2


# BACKBONE = 'densenet169'
BACKBONE = 'ResNext101_64x4d'
CKPT_FOLD = "ckpt_ddt1"
# CKPT_NAME = "efficientnetb0_img224224_demo"
CKPT_NAME = BACKBONE + "_img" + str(IMG_SIZE[0]) + str(IMG_SIZE[1]) + "_b" + str(TRAIN_BATCH_SIZE)
CKPT_PATH = f"./{CKPT_FOLD}/{CKPT_NAME}"

TRAIN_VALID_FLAG = True
TEST_FLAG = True
# 预训练模型路径
PRETRAINED_MODEL_DIR = r"E:\天池\ICDAR文本篡改分类和检测\prertrained_checkpoint"

# Data Augmentations
def build_transforms():
    data_transforms = {
        "train": A.Compose([
            A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1], p=1.0),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomGridShuffle(grid=(3,3),p=0.2),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.HueSaturationValue(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=1),
            ], p=0.3),
            A.GaussNoise(p=0.25),
            A.OneOf([
                A.MotionBlur(p=1),
                A.GaussianBlur(p=1),
                A.ImageCompression(quality_lower=65, quality_upper=80, p=1),
            ], p=0.3),
            # A.Normalize(
            #     mean=[0.3199, 0.2240, 0.1609],
            #     std=[0.3020, 0.2183, 0.1741],
            #     max_pixel_value=255.0,
            # ),
        ], p=1.0),

        "valid_test": A.Compose([
            A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1],interpolation=cv2.INTER_NEAREST, p=1.0),
            # A.Normalize(
            #     mean=[0.3199, 0.2240, 0.1609],
            #     std=[0.3020, 0.2183, 0.1741],
            #     max_pixel_value=255.0,
            # ),
        ], p=1.0)
    }
    return data_transforms

# # Data Augmentations
# def build_transforms():
#     data_transforms = {
#         "train": A.Compose([
#             A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1], p=1.0),
#             A.RandomRotate90(p=0.5),
#             A.Rotate(limit=30, p=0.3),
#             A.ColorJitter(p=0.3),
#             A.CoarseDropout(max_holes=6, max_height=5, max_width=5, p=0.3),
#             A.GaussianBlur(p=0.3),
#             A.GaussNoise(p=0.3),
#             A.OneOf([
#                 # A.IAAAdditiveGaussianNoise(p=0.3),
#                 A.RandomBrightnessContrast(p=0.1),
#                 A.HueSaturationValue(p=0.5),
#                 A.ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
#                 #A.CoarseDropout(p=0.2),
#                 A.Transpose(p=0.2),
#                 A.Blur(blur_limit=3, p=0.2),
#                 A.MedianBlur(blur_limit=3, p=0.2),
#                 A.MotionBlur(p=0.2),
#                 A.GridDistortion(p=0.2),
#                 A.RandomFog(p=0.2),
#             ]),
#             A.Normalize(
#                 mean=[0.3199, 0.2240, 0.1609],
#                 std=[0.3020, 0.2183, 0.1741],
#                 max_pixel_value=255.0,
#             ),
#         ], p=1.0),
#
#         "valid_test": A.Compose([
#             A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1],interpolation=cv2.INTER_NEAREST, p=1.0),
#             A.Normalize(
#                 mean=[0.3199, 0.2240, 0.1609],
#                 std=[0.3020, 0.2183, 0.1741],
#                 max_pixel_value=255.0,
#             ),
#         ], p=1.0)
#     }
#     return data_transforms

def main():
    x = torch.rand([1, 3, 512, 512])
    data_transforms = build_transforms()


if __name__ == "__main__":
    main()