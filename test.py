from modeling.deeplab import *
from paraseres import ad1para
from AD1Dataloader import ad1loader
from metrics import Evaluator
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import cv2
import torchvision.transforms as transforms
import h5py

if __name__ == "__main__":

    train_dir = 'E:/lab2_train_data.h5'
    test_dir = 'E:/lab2_test_data.h5'
    data = h5py.File(train_dir, 'r')
    color = data['color_codes'][:]

    args = ad1para().parse_args()
    num_classes = args.numclasses
    model = DeepLab(output_stride=args.output_stride, num_classes=num_classes)

    model.load_state_dict(torch.load(args.premodel))
    model.cuda()
    train_loader, val_loader = ad1loader(train_path=args.train_path, val_path=args.val_path,
                                         trainbatch_size=args.train_batch, testbatch_size=args.val_batch)

    for step, (image, target) in enumerate(val_loader):
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        target = target.permute(0, 2, 3, 1) * 255
        target = torch.squeeze(target)

        with torch.no_grad():
            output = model(image)

        # images = image[1, :, :, :].cpu()
        # label = target[1, :, :, :].cpu()
        # result = output[1, :, :, :].cpu()
        #
        # images = images.permute(0, 1, 2)
        # cv2.imshow('images', images.numpy())
        #
        # label = label.permute(1, 0, 2)
        # cv2.imshow('label', label.numpy())
        #
        # result = result.permute(1, 0, 2)
        # cv2.result('result', result.numpy())
        #
        # cv2.waitKey(5000)

        bibi = torch.max(output, 1)[1]
        bibi = bibi.cpu()
        lili = image.cpu()
        tata = target.cpu()
        for m in range(32):
            result = bibi[m, :, :]
            img = lili[m, :, :, :]
            img = img.permute(1, 2, 0)
            lab = tata[m, :, :]
            lab = lab.type(torch.int64)
            h, w = result.shape[0], result.shape[1]
            result_rgb = np.zeros((h, w, 3))
            lab_rgb = np.zeros((h, w, 3))

            for i, rgb in zip(range(34), color):
                # print(i,rgb) # 数字对应颜色
                result_rgb[result[:, :] == i] = rgb
            for li, lrgb in zip(range(34), color):
                # print(i,rgb) # 数字对应颜色
                lab_rgb[lab[:, :] == li] = lrgb
            result_rgb = result_rgb.astype(np.uint8)
            lab_rgb = lab_rgb.astype(np.uint8)
            # cv2.imshow('lab_rgb', lab_rgb)

            multi_out = (result_rgb/255)*0.3 + (img.numpy()*0.5 + 0.5)*0.7
            multi_tar = (lab_rgb / 255) * 0.3 + (img.numpy() * 0.5 + 0.5) * 0.7
            out_tar = np.hstack((multi_out, multi_tar))*255

            save_name = 'E:/AD1/out/' + str(step*32+m) + '.png'
            cv2.imwrite(save_name, out_tar)
            # cv2.imshow('out_tar', out_tar)
            # cv2.waitKey(500)
        print('nice')

