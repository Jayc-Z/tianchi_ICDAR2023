# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 21:59
# @Author  : Curry
# @File    : utils.py

import config
import random
import torch
import numpy as np
import os
import pandas as pd

def set_seed(seed=config.SEED):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed) # python
    np.random.seed(seed) # numpy
    torch.manual_seed(seed) # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def build_train_valid_df():
    col_name = ['img_name', 'img_path', 'img_label']
    imgs_info = []  # img_name, img_path, img_label
    for img_name in os.listdir(config.TAMPERED_IMG_PATH):
        if img_name.endswith('.jpg'):  # pass other files
            imgs_info.append(["p_" + img_name, os.path.join(config.TAMPERED_IMG_PATH, img_name), 1])

    for img_name in os.listdir(config.UNTAMPERED_IMG_PATH):
        if img_name.endswith('.jpg'):  # pass other files
            imgs_info.append(["n_" + img_name, os.path.join(config.UNTAMPERED_IMG_PATH, img_name), 0])

    imgs_info_array = np.array(imgs_info)
    df = pd.DataFrame(imgs_info_array, columns=col_name)
    return df

def build_test_df():
    col_name = ['img_name', 'img_path', 'pred_prob']
    imgs_info = []
    test_imgs = os.listdir(config.TEST_IMG_PATH)
    test_imgs.sort(key=lambda x: x[:-4])
    for img_name in test_imgs:
        if img_name.endswith('.jpg'):  # pass other files
            imgs_info.append([img_name, os.path.join(config.TEST_IMG_PATH, img_name), 0])

    imgs_info_array = np.array(imgs_info)
    test_df = pd.DataFrame(imgs_info_array, columns=col_name)

