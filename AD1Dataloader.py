from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from AD1Dataset import AD1Dataset


def ad1loader(train_path, val_path, trainbatch_size=16, testbatch_size=16):

    transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ]
    )

    labels_transform = transforms.Compose(
        transforms=[
            transforms.ToTensor()
        ]
    )

    train_set = AD1Dataset(train_path, transform=transform, labels_transform=labels_transform)
    train_loader = DataLoader(dataset=train_set, batch_size=trainbatch_size, shuffle=True, num_workers=0)
    val_set = AD1Dataset(val_path,  transform=transform, labels_transform=labels_transform)
    val_loader = DataLoader(dataset=val_set, batch_size=testbatch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def ad1loaderhf(train_path, val_path, trainbatch_size=16, testbatch_size=16):
    transform = transforms.Compose(
        transforms=[
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ]
    )

    labels_transform = transforms.Compose(
        transforms=[
            transforms.ToTensor()

        ]
    )

    train_set = AD1Dataset(train_path, transform=transform, labels_transform=labels_transform, flip=1)
    train_loader = DataLoader(dataset=train_set, batch_size=trainbatch_size, shuffle=True, num_workers=0)
    val_set = AD1Dataset(val_path,  transform=transform, labels_transform=labels_transform, flip=1)
    val_loader = DataLoader(dataset=val_set, batch_size=testbatch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader

def ad1loadervf(train_path, val_path, trainbatch_size=16, testbatch_size=16):
    transform = transforms.Compose(
        transforms=[
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ]
    )

    labels_transform = transforms.Compose(
        transforms=[
            transforms.ToTensor()
        ]
    )

    train_set = AD1Dataset(train_path, transform=transform, labels_transform=labels_transform, flip=0)
    train_loader = DataLoader(dataset=train_set, batch_size=trainbatch_size, shuffle=True, num_workers=0)
    val_set = AD1Dataset(val_path,  transform=transform, labels_transform=labels_transform, flip=0)
    val_loader = DataLoader(dataset=val_set, batch_size=testbatch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader