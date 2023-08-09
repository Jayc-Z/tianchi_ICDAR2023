# -*- coding: utf-8 -*-
# @Time    : 2023/3/6 22:31
# @Author  : Curry
# @File    : train_stacking.py

import torch
from sklearn.model_selection import KFold
from dataset import build_dataset, build_dataloader
import config
from config import build_transforms
from utils import set_seed, mkdirs, build_test_df, build_train_valid_df
import os
from model import build_model
from loss import build_loss, BCEFocalLoss, focal_lossv1
from tqdm import tqdm
from torch.cuda import amp
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
from metric import build_score_cls

os.environ['TORCH_HOME'] = config.PRETRAINED_MODEL_DIR
torch.set_num_threads(4)  # 限制占用核心数最大为4
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_one_epoch(model, train_loader, optimizer, losses_dict):
    model.train()
    scaler = amp.GradScaler()
    losses_all, ce_all = 0, 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    for _, (images, gt) in pbar:
        optimizer.zero_grad()

        images = images.to(config.DEVICE, dtype=torch.float)  # [b, c, w, h]

        gt = gt.to(config.DEVICE)

        with amp.autocast(enabled=True):
            y_preds = model(images)
            ce_loss = losses_dict["CELoss"](y_preds, gt.long())
            focalloss = focal_lossv1(y_preds, gt.long())

            # target = F.one_hot(gt.to(torch.int64), config.NUM_CLASSES).float()
            # weight_bceloss = F.binary_cross_entropy_with_logits(y_preds, target, pos_weight=torch.tensor(15))
            # losses = ce_loss + focalloss + weight_bceloss
            losses = focalloss

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        losses_all += losses.item() / images.shape[0]
        ce_all += ce_loss.item() / images.shape[0]

    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, ce_all: {:.3f}".format(losses_all, ce_all), flush=True)


@torch.no_grad()
def valid_one_epoch(model, valid_loader, valid_img_name):
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    y_pred_array = np.array([])
    label_array = np.array([])
    valid_col_name = ['img_name', 'img_path', "valid_pred_prob", "valid_pred_class"]
    for _, (images, gt) in pbar:
        images = images.to(config.DEVICE, dtype=torch.float)  # [b, c, w, h]
        gt = gt.to(config.DEVICE)
        label_array = np.concatenate((label_array, gt.cpu().numpy()), axis=0)
        y_preds = model(images)

        prob = F.softmax(y_preds, dim=1).detach().cpu().numpy()
        y_preds_value, y_preds = torch.max(y_preds.data, dim=1)

        y_pred_array = np.concatenate((y_pred_array, prob[:, 1]), axis=0)
        correct += (y_preds == gt).sum()
        total += gt.shape[0]

    pred_dict = {"0": valid_img_name, "1": y_pred_array}
    label_dict = {"0": valid_img_name, "1": label_array}
    pred_df = pd.DataFrame(pred_dict)
    label_df = pd.DataFrame(label_dict)
    val_acc = correct / total
    print("val_acc: {:.2f}".format(val_acc), flush=True)

    return val_acc, pred_df, label_df


def main():
    set_seed(config.SEED)
    mkdirs(config.CKPT_PATH)
    if config.TRAIN_VALID_FLAG:
        df = build_train_valid_df()
        # skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        kf = KFold(n_splits=config.N_FOLD, shuffle=True, random_state=config.SEED)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold
        for fold in range(config.N_FOLD):
            print(f'#' * 40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#' * 40, flush=True)

            data_transforms = build_transforms()
            train_loader, valid_loader = build_dataloader(df, config.TRAIN_VALID_FLAG, fold,
                                                          data_transforms)  # dataset & dtaloader
            model = build_model(pretrain_flag=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
            losses_dict = build_loss()  # loss

            valid_img_name = df["img_name"].loc[df["fold"] == fold]

            best_val_acc = 0
            best_epoch = 0
            best_score = 0

            for epoch in range(1, config.NUM_EPOCH + 1):
                start_time = time.time()
                train_one_epoch(model, train_loader, optimizer, losses_dict)
                lr_scheduler.step()
                val_acc, pred_df, label_df = valid_one_epoch(model, valid_loader, valid_img_name)
                score = build_score_cls(pred_df, label_df)
                print("val_score: {:.2f}".format(score), flush=True)

                # is_best = (val_acc > best_val_acc)
                is_best = (score >= best_score)
                best_score = max(score, best_score)
                best_val_acc = max(best_val_acc, val_acc)
                if is_best:
                    save_path = f"{config.CKPT_PATH}/best_fold{fold}_epoch{epoch}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path)
                    torch.save(model.state_dict(), save_path)

                    # stacking保存验证集预测结果
                    save_valid_prob_path = f"{config.CKPT_PATH}/best_fold{fold}_epoch{epoch}_prob.csv"
                    save_valid_label_path = f"{config.CKPT_PATH}/best_fold{fold}_epoch{epoch}_label.csv"

                    pred_df.to_csv(save_valid_prob_path, header=False, index=False, sep=' ')
                    label_df.to_csv(save_valid_label_path, header=False, index=False, sep=' ')

                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best_acc:{:.2f}, best_score:{:.2f}\n".format
                      (epoch, epoch_time, best_val_acc, best_score), flush=True)


if __name__ == "__main__":
    main()