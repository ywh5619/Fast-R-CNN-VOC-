from modeling.deeplab import *
from paraseres import ad1para
from AD1Dataloader import *
from metrics import Evaluator
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import cv2
import h5py

if __name__ == "__main__":


    train_dir = 'E:/lab2_train_data.h5'
    test_dir = 'E:/lab2_test_data.h5'
    data = h5py.File(train_dir, 'r')
    color = data['color_codes'][:]


    args = ad1para().parse_args()

    torch.manual_seed(args.seed)
    num_classes = args.numclasses

    train_loader, val_loader = ad1loader(train_path=args.train_path, val_path=args.val_path,
                                         trainbatch_size=args.train_batch, testbatch_size=args.val_batch)
    train_loadervf, val_loadervf = ad1loadervf(train_path=args.train_path, val_path=args.val_path,
                                         trainbatch_size=args.train_batch, testbatch_size=args.val_batch)
    train_loaderhf, val_loaderhf = ad1loaderhf(train_path=args.train_path, val_path=args.val_path,
                                         trainbatch_size=args.train_batch, testbatch_size=args.val_batch)

    for i, (batch_x, batch_y) in enumerate(train_loadervf):
        for i in range(100):
            image = batch_x[i, :, :, :]
            label_org = batch_y[i, :, :, :]
            image = image.permute(1, 2, 0)
            label_org = label_org.permute(1, 2, 0) * 255
            print(image.shape)
            cv2.imshow('image', image.numpy())
            h, w = label_org.shape[0], label_org.shape[1]
            label = np.zeros((h, w, 3))
            for i, rgb in zip(range(34), color):
                # print(i,rgb) # number maps color
                label[label_org[:, :, 0] == i] = rgb
            label = label.astype(np.uint8)
            cv2.imshow('label', label)

            cv2.waitKey(0)