# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 22:59
# @Author  : Curry
# @File    : inference.py

import config
import os
import numpy as np
import pandas as pd
from dataset import build_dataloader
from config import build_transforms
from tqdm import tqdm
from model import build_model
import torch

@torch.no_grad()
def eval_one_epoch(ckpt_paths, test_loader, test_df):

    print("Inference models:" + "\n" + str([m+"\n" for m in ckpt_paths]))
    num_models = len(ckpt_paths)
    for model_index, sub_ckpt_path in enumerate(ckpt_paths):
        model_id = model_index + 1
        print(f"Loading model {model_id}:{sub_ckpt_path}")
        # model = eval_build_model()
        model = build_model(pretrain_flag=False)
        model.load_state_dict(torch.load(sub_ckpt_path), strict=False)
        model.eval()
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
        for _, (ids, images) in enumerate(pbar):
            images = images[0].to(config.DEVICE, dtype=torch.float)
            y_preds = model(images)
            # 预测概率
            prob = torch.nn.functional.softmax(y_preds, dim=-1)[:, 1].detach().cpu().numpy()
            prod_index = f'pred_prob{model_id}'
            test_df.loc[test_df.index == ids, prod_index] = prob
            # 预测类别
            _, pred_class = torch.max(y_preds.data, dim=1)
            class_index = f"pred_class{model_id}"
            pred_class = pred_class.detach().cpu().numpy()
            test_df.loc[test_df.index == ids, class_index] = pred_class

    test_df["avg_prob"] = test_df[['pred_prob1', 'pred_prob2', 'pred_prob3', 'pred_prob4', 'pred_prob5']].mean(axis=1)
    return test_df

def get_ckpt_paths(models_root_dir):
    ckpt_info = []
    ckpt_list = os.listdir(models_root_dir)
    for ckpt in ckpt_list:
        if ckpt.endswith('.pth'):
            ckpt_info.append(os.path.join(models_root_dir, ckpt))
    return ckpt_info

def eval():
    if config.TEST_FLAG:
        # 权重根目录
        weights_root_dir = "../src"
        # 子模型文件夹
        sub_weights_list = os.listdir(weights_root_dir)
        for sub_weights_dir in sub_weights_list:
            if os.path.isdir(sub_weights_dir):
                ckpt_paths = get_ckpt_paths(sub_weights_dir)
                col_name = ['img_name', 'img_path', "avg_prob"]
                for i in range(len(ckpt_paths)):
                    pred_prob = f"pred_prob{i+1}"
                    col_name.append(pred_prob)

        imgs_info = []
        test_imgs = os.listdir(config.TEST_IMG_PATH)
        test_imgs.sort(key=lambda x: x[:-4])
        for img_name in test_imgs:
            if img_name.endswith('.jpg'):  # pass other files
                title_info = [img_name, os.path.join(config.TEST_IMG_PATH, img_name), 0, 0]
                for i in range(len(ckpt_paths)):
                    title_info.append(0)
                    title_info.append(0)
                imgs_info.append(title_info)
        imgs_info_array = np.array(imgs_info)
        test_df = pd.DataFrame(imgs_info_array, columns=col_name)

        data_transforms = build_transforms()
        test_loader = build_dataloader(test_df, False, None, data_transforms)  # dataset & dtaloader

        # test
        test_df = eval_one_epoch(ckpt_paths, test_loader, test_df)

        submit_df = test_df.loc[:, ['img_name', 'avg_prob']]
        submit_df.to_csv(os.path.join(models_root_dir,"submit_dummy.csv"), header=False, index=False, sep=' ')

if __name__ == "__main__":
    eval()