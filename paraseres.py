import argparse


def ad1para():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

    parser.add_argument('--train_path', type=str, default='E:/lab2_train_data.h5',
                        help='the path of train data')
    parser.add_argument('--val_path', type=str, default='E:/lab2_test_data.h5',
                        help='the path of Validate data')
    parser.add_argument('--premodel', type=str, default='I_am_Conv_King.pkl',
                        help='the path of Pretrain model')
    # training hyper params
    parser.add_argument('--numclasses', type=int, default=34,
                        help='number of class of seg (default: 34)')
    parser.add_argument('--epoch', type=int, default=256,
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epochs (default:0)')
    parser.add_argument('--train_batch', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val_batch', type=int, default=32,
                        help='input batch size for val (default: 32)')
    parser.add_argument('--output_stride', type=int, default=8,
                        help='network output stride (default: 8)')

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='gamma of StepLR (default: 0.95)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    return parser
