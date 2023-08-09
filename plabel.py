# -*- coding: utf-8 -*-
# @Time    : 2023/3/6 16:38
# @Author  : Curry
# @File    : plabel.py

import pandas as pd
import shutil
import os
import config

def build_vote_label():
    test_label_file = "efficientnet_b7_img512512_bs8_adddata/test_label.csv"
    test_label_df = pd.read_csv(test_label_file, sep=" ")

    test_label_df["vote_class"] = test_label_df[['pred_class1', 'pred_class2', 'pred_class3', 'pred_class4']].mean(axis=1)
    count = 0
    for index,row in test_label_df.iterrows():
        test_label_df.loc[index, "vote_class"] = 1 if row["vote_class"] > 0 else 0
        if row["vote_class"] == 0:
            count += 1
    test_label_df[["img", "vote_class"]].to_csv("efficientnet_b7_img512512_bs8_adddata/vote_label1.csv", index=False)
    print(count)

def copy_img(local_img_path, save_dir):
    save_dir_imgs_list = os.listdir(save_dir)
    len_save_dir = len(save_dir_imgs_list)
    if local_img_path.endswith('.jpg'):
        img_name = local_img_path[-8:]
        if img_name in save_dir_imgs_list:
            img_name = str(len_save_dir) + ".jpg"
        save_path = os.path.join(save_dir, img_name)
        shutil.copy(local_img_path, save_path)
        print(f"copy {img_name} to f{save_path}!")

def get_plabel_data(vote_label_file, tampered_path, untampered_path):
    vote_label_df = pd.read_csv(vote_label_file, sep=",")
    for index, row in vote_label_df.iterrows():
        img_name, label = row["img"], int(row["vote_class"])
        img_path = os.path.join(config.TEST_IMG_PATH, img_name)
        if label == 1:
            copy_img(img_path, tampered_path)
        else:
            copy_img(img_path, untampered_path)


vote_label_file = "efficientnet_b7_img512512_bs8_adddata/vote_label1.csv"
tampered_path, untampered_path = config.TAMPERED_IMG_PATH, config.UNTAMPERED_IMG_PATH
get_plabel_data(vote_label_file, tampered_path, untampered_path)