# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 23:04
# @Author  : Curry
# @File    : model_swin.py
# -*- coding: UTF-8 -*-

from sklearn.model_selection import GroupKFold, StratifiedKFold
import torch
from torch import nn
import os
import time
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import timm
import cv2

# train_csv_path = '../train0820.csv'  # 数据集train.csv路径
# train_img_path = '../train0820/'  # 训练集图片路径
CFG = {
    'fold_num': 5,
    'seed': 719,
    'model_arch': 'swin_large_patch4_window7_224',
    'img_size': 448,
    'epochs': 15,
    'train_bs': 2,
    'valid_bs': 2,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0',
    'classnum': 3
}
# train = pd.read_csv(train_csv_path)
# train.head()
# train.label.value_counts()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    '''使用 opencv 加载图片.
    由于历史原因，opencv 读取的图片格式是 bgr
    Args:
        path : str  图片文件路径 e.g '../data/train_img/1.jpg'
    '''
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(path)
        paths = path.split('.')
        path = paths[0] + '.png'
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self, df, data_root,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if output_label == True:
            self.labels = self.df['label'].values
            # print(self.labels)

            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max() + 1)[self.labels]
                # print(self.labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]

        img = get_img("{}/{}".format(self.data_root + str(self.df.loc[index]['label']), self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.output_label == True:
            return img, target
        else:
            return img


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        #         HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        #         RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        #         CoarseDropout(p=0.5),
        #         Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms():
    return Compose([
        Resize(CFG['img_size'], CFG['img_size']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


class myModel(nn.Module):
    def __init__(self,
                 arch_name,
                 pretrained=False,
                 img_size=256,
                 multi_drop=False,
                 multi_drop_rate=0.5,
                 att_layer=False,
                 att_pattern="A"
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.att_layer = att_layer
        self.multi_drop = multi_drop

        self.model = timm.create_model(
            arch_name, pretrained=pretrained
        )
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()

        self.head = nn.Linear(n_features, 3)
        self.head_drops = nn.ModuleList()
        for i in range(3):
            self.head_drops.append(nn.Dropout(multi_drop_rate))

        if att_layer:
            if att_pattern == "A":
                self.att_layer = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
            elif att_pattern == "B":
                self.att_layer = nn.Linear(n_features, 1)
            else:
                raise ValueError("invalid att pattern")

    def forward(self, x):
        if self.att_layer:
            l = x.shape[2] // 2
            h1 = self.model(x[:, :, :l, :l])
            h2 = self.model(x[:, :, :l, l:])
            h3 = self.model(x[:, :, l:, :l])
            h4 = self.model(x[:, :, l:, l:])
            w = F.softmax(torch.cat([
                self.att_layer(h1),
                self.att_layer(h2),
                self.att_layer(h3),
                self.att_layer(h4),
            ], dim=1), dim=1)
            h = h1 * w[:, 0].unsqueeze(-1) + \
                h2 * w[:, 1].unsqueeze(-1) + \
                h3 * w[:, 2].unsqueeze(-1) + \
                h4 * w[:, 3].unsqueeze(-1)
        else:
            h = self.model(x)

        if self.multi_drop:
            for i, dropout in enumerate(self.head_drops):
                if i == 0:
                    output = self.head(dropout(h))
                else:
                    output += self.head(dropout(h))
            output /= len(self.head_drops)
        else:
            output = self.head(h)
        return output


class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, n_class)

        #         n_features = self.model.head.fc.in_features
        #         self.model.head.fc = nn.Linear(n_features, n_class)
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''

    def forward(self, x):
        x = self.model(x)
        return x


def prepare_dataloader(df, trn_idx, val_idx, data_root=r'../data/train'):
    # from catalyst.data.sampler import BalanceClassSampler

    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    train_ds = CassavaDataset(train_, data_root, transforms=get_train_transforms(), output_label=True,
                              one_hot_label=False)
    valid_ds = CassavaDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
        # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        # print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)  # output = model(input)

            loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'

                pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, fold, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum / sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all == image_targets_all).mean()))
    acc = (image_preds_all == image_targets_all).mean()
    if acc > 0.79:  # 可根据实际情况调整阈值
        torch.save(model.state_dict(), 'result/{}_fold_{}_{}_{}'.format(CFG['model_arch'], fold, epoch, acc))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()


class FocalLoss:
    def __init__(self, alpha_t=None, gamma=0):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def __call__(self, outputs, targets):
        if self.alpha_t is None and self.gamma == 0:
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets).to(device)

        elif self.alpha_t is not None and self.gamma == 0:
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                           weight=self.alpha_t)

        elif self.alpha_t is None and self.gamma != 0:
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

        elif self.alpha_t is not None and self.gamma != 0:
            device = torch.device(CFG['device'])
            if self.alpha_t.device != outputs.device:
                self.alpha_t = self.alpha_t.to(outputs)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none').to(device)
            p_t = torch.exp(-ce_loss)
            ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
                                                        weight=self.alpha_t, reduction='none').to(device)
            focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss


if __name__ == '__main__':
    # for training only, need nightly build pytorch

    seed_everything(CFG['seed'])

    # folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(
    #     np.arange(train.shape[0]), train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        #         if fold == 0 or fold == 1 or fold ==2  :
        #             continue
        # we'll train fold 0 first
        print('Training with {} started'.format(fold))

        print(len(trn_idx), len(val_idx))
        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx,
                                                      data_root=train_img_path)

        device = torch.device(CFG['device'])

        # model = CassvaImgClassifier(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)
        model = myModel(arch_name=CFG['model_arch'],
                        pretrained=True,
                        img_size=CFG['img_size'],
                        multi_drop=False,
                        multi_drop_rate=0.5,
                        att_layer=True,
                        att_pattern="A").to(device)
        model = nn.DataParallel(model, device_ids=[0])
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1,
                                                                         eta_min=CFG['min_lr'], last_epoch=-1)

        # loss_tr = nn.CrossEntropyLoss().to(device)
        # loss_fn = nn.CrossEntropyLoss().to(device)
        loss_tr = FocalLoss([0.33, 0.33, 0.34], 2)
        loss_fn = FocalLoss([0.33, 0.33, 0.34], 2)

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler,
                            schd_batch_update=False)

            with torch.no_grad():
                valid_one_epoch(epoch, model, loss_fn, val_loader, device, fold, scheduler=None, schd_loss_update=False)

        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()