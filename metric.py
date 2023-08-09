# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 23:20
# @Author  : Curry
# @File    : metric.py

import numpy as np

def build_score_cls(pred_df, label_df):
    pred_df = pred_df.values
    label_df = label_df.values
    tampers = label_df[label_df[:, 1] == 1]
    untampers = label_df[label_df[:, 1] == 0]
    pred_tampers = pred_df[np.in1d(pred_df[:, 0], tampers[:, 0])]
    pred_untampers = pred_df[np.in1d(pred_df[:, 0], untampers[:, 0])]
    thres = np.percentile(pred_untampers[:, 1], np.arange(90, 100, 1))  # np.percentile求分位数
    # np.greater判断前者是否比后者大, np.newaxis生成新维度
    recall = np.mean(np.greater(pred_tampers[:, 1][:, np.newaxis], thres).mean(axis=0))
    return recall * 100