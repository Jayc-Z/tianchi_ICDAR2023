# -*- coding: utf-8 -*-
# @Time    : 2023/3/8 0:39
# @Author  : Curry
# @File    : model_stacking.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, BayesianRidge
import os

def get_train_oof(model_path):
    prob_csv_list = []
    for file in os.listdir(model_path):
        if "prob" in file:
            prob_csv_list.append(file)
    train_oof1 = pd.read_csv(os.path.join(model_path, prob_csv_list[0]), header=None, sep=" ")
    train_oof2 = pd.read_csv(os.path.join(model_path, prob_csv_list[1]), header=None, sep=" ")
    train_oof3 = pd.read_csv(os.path.join(model_path, prob_csv_list[2]), header=None, sep=" ")
    train_oof4 = pd.read_csv(os.path.join(model_path, prob_csv_list[3]), header=None, sep=" ")
    train_oof = pd.concat([train_oof1, train_oof2, train_oof3, train_oof4], axis=0,)
    train_oof.columns = ["img", "prob"]
    train_oof = train_oof.sort_values(by="img", ascending=True)
    train_oof = train_oof.reset_index(drop=True)
    return train_oof

def get_test_predict(model_path):
    submit_dummy = None
    model_test_predict = None
    for file in os.listdir(model_path):
        if "submit_dummy" in file:
            submit_dummy = file
            model_test_predict = pd.read_csv(os.path.join(model_path, submit_dummy), header=None, sep=" ")
            model_test_predict.columns = ["img", "prob"]

    return model_test_predict

# model1_path = ""
model2_path = "ConvNext_img512512_b8_V4"
model3_path = "efficientnet_b7_img512512_bs8_V4"
model4_path = "Densenet201_img512512_b32_V4"

# train_oof1 = get_train_oof(model1_path)
train_oof2 = get_train_oof(model2_path)
train_oof3 = get_train_oof(model3_path)
train_oof4 = get_train_oof(model4_path)
# train_oof1.rename(columns={'prob': 'VGG16_oof'}, inplace=True)
train_oof2.rename(columns={'prob': 'resnet152_oof'}, inplace=True)
train_oof3.rename(columns={'prob': 'efficientnetb7_oof'}, inplace=True)
train_oof4.rename(columns={'prob': 'DenseNet169_oof'}, inplace=True)
train_oof = pd.concat([train_oof2,
                       # train_oof2["resnet152_oof"],
                       train_oof3["efficientnetb7_oof"],
                       train_oof4["DenseNet169_oof"]
                       ], axis=1)

y_train_label = np.zeros(train_oof["img"].shape[0])
y_train = pd.DataFrame({"img":train_oof["img"],
                        "y_train_label":y_train_label})

for index, row in y_train.iterrows():
    img = row["img"]
    # print(img)
    if "n" in img:
        y_train_label = 0
    else:
        y_train_label = 1
    y_train.loc[index, "y_train_label"] = y_train_label


#测试集预测各模型预测结果
# VGG16_test_predict = get_test_predict(model1_path)
resnet152_test_predict = get_test_predict(model2_path)
efficientnetb7_test_predict = get_test_predict(model3_path)
DenseNet169_test_predict = get_test_predict(model4_path)

# VGG16_test_predict.rename(columns={'prob': 'VGG16_oof'}, inplace=True)
resnet152_test_predict.rename(columns={'prob': 'resnet152_oof'}, inplace=True)
efficientnetb7_test_predict.rename(columns={'prob': 'efficientnetb7_oof'}, inplace=True)
DenseNet169_test_predict.rename(columns={'prob': 'DenseNet169_oof'}, inplace=True)
test_predict = pd.concat([
                        resnet152_test_predict,
                       # resnet152_test_predict["resnet152_oof"],
                       efficientnetb7_test_predict["efficientnetb7_oof"],
                       DenseNet169_test_predict["DenseNet169_oof"]
                            ], axis=1)

LR = LogisticRegression().fit(train_oof[["resnet152_oof",
                                         # "resnet152_oof",
                                         "efficientnetb7_oof",
                                         "DenseNet169_oof"
                                         ]
                                            ], y_train["y_train_label"])
train_score = LR.score(train_oof[["resnet152_oof",
                                  # "resnet152_oof",
                                  "efficientnetb7_oof",
                                  "DenseNet169_oof"
                                  ]], y_train["y_train_label"])

predicts = LR.predict_proba(test_predict[["resnet152_oof",
                                  # "resnet152_oof",
                                  "efficientnetb7_oof",
                                  "DenseNet169_oof"
                                          ]])
predicts = pd.DataFrame(predicts)
pred = predicts[1]
final_df = pd.DataFrame({"img":test_predict["img"],
                         "prob":pred})
save_dir = 'efficientb7_densenet201_ConvNextL_stacking_V4'
final_df.to_csv(os.path.join(save_dir, "submit_dummy_82.82.csv"), header=None, sep=" ", index=False)