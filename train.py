from modeling.deeplab import *
from paraseres import ad1para
from AD1Dataloader import *
from metrics import Evaluator
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":

    args = ad1para().parse_args()

    torch.manual_seed(args.seed)
    num_classes = args.numclasses

    train_loader, val_loader = ad1loader(train_path=args.train_path, val_path=args.val_path,
                                         trainbatch_size=args.train_batch, testbatch_size=args.val_batch)
    train_loadervf, val_loadervf = ad1loadervf(train_path=args.train_path, val_path=args.val_path,
                                         trainbatch_size=args.train_batch, testbatch_size=args.val_batch)
    train_loaderhf, val_loaderhf = ad1loaderhf(train_path=args.train_path, val_path=args.val_path,
                                         trainbatch_size=args.train_batch, testbatch_size=args.val_batch)

    # Define network
    model = DeepLab(output_stride=args.output_stride, num_classes=num_classes)

    train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                    {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

    # Define Optimizer
    optimizer = optim.SGD(train_params, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=args.gamma)

    loss_func = nn.CrossEntropyLoss()

    evaluator = Evaluator(num_classes)

    if args.premodel is not None:
        model.load_state_dict(torch.load(args.premodel))
    if args.cuda:
        model = model.cuda()
    model.train()
    Max_MIOU = 0
    Max_Acc = 0
    PM = 0
    for epoch in range(args.epoch):
        train_loss = 0
        train_lossvf = 0
        train_losshf = 0

        # tbarvf = tqdm(train_loadervf)
        # num_img_tr = len(train_loadervf)
        # for i, (image, target) in enumerate(tbarvf):
        #     if args.cuda:
        #         image, target = image.cuda(), target.cuda()
        #     target = target.permute(0, 2, 3, 1) * 255
        #     target = torch.squeeze(target)
        #
        #     optimizer.zero_grad()
        #     output = model(image)
        #     loss = loss_func(output, target.long())
        #     loss.backward()
        #     optimizer.step()
        #     train_lossvf += loss.item()
        #     tbarvf.set_description(f'Trainvf loss:{train_lossvf / (i + 1)}')

        tbar = tqdm(train_loader)
        num_img_tr = len(train_loader)
        for i, (image, target) in enumerate(tbar):
            if args.cuda:
                image, target = image.cuda(), target.cuda()
            target = target.permute(0, 2, 3, 1) * 255
            target = torch.squeeze(target)

            optimizer.zero_grad()
            output = model(image)
            loss = loss_func(output, target.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            tbar.set_description(f'Train loss:{train_loss / (i + 1)}')


        tbarhf = tqdm(train_loaderhf)
        num_img_tr = len(train_loaderhf)
        for i, (image, target) in enumerate(tbarhf):
            if args.cuda:
                image, target = image.cuda(), target.cuda()
            target = target.permute(0, 2, 3, 1) * 255
            target = torch.squeeze(target)

            optimizer.zero_grad()
            output = model(image)
            loss = loss_func(output, target.long())
            loss.backward()
            optimizer.step()
            train_losshf += loss.item()
            tbarhf.set_description(f'Trainhf loss:{train_losshf / (i + 1)}')

        print(f"train Loss: {(train_loss + train_losshf )/2} | leaning rate: {optimizer.param_groups[0]['lr']}")


        # Validation
        model.eval()
        evaluator.reset()
        tbar = tqdm(val_loader, desc='\r')
        test_loss = 0.0

        for i, (image, target) in enumerate(tbar):
            if args.cuda:
                image, target = image.cuda(), target.cuda()
            target = target.permute(0, 2, 3, 1) * 255
            target = torch.squeeze(target)

            with torch.no_grad():
                output = model(image)
            loss = loss_func(output, target.long())
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        if epoch == 0:
            Max_MIOU = mIoU
            Max_Acc = Acc
        if (mIoU > Max_MIOU) & (Acc > Max_Acc):
            # net_params = 'Acc_' + str(Acc) + 'MIoU_' + str(mIoU) + '.pkl'
            net_params = 'I_am_Conv_King.pkl'
            torch.save(model.state_dict(), net_params)  # only save params
            Max_MIOU = mIoU
            Max_Acc = Acc

        print(f'Epoch{epoch}-Validation-Loss:{test_loss} | best_MIou:{Max_MIOU}')
        print(f"Acc_pixel:{Acc} | Acc_class:{Acc_class}")
        print(f"Mean_IoU:{mIoU} | FWIoU:{FWIoU}")

        if mIoU > PM:
            Counter = 0
        else:
            Counter += 1
        if Counter > 2:
            scheduler.step()
            if Counter > 8:
                break
        PM = mIoU

