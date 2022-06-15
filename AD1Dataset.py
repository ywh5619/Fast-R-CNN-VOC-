from torch.utils.data.dataset import Dataset
import h5py
from PIL import Image
import cv2

class AD1Dataset(Dataset):
    def __init__(self, root, transform=None, labels_transform=None, flip=2):
        self.data = h5py.File(root, 'r')
        self.transform = transform
        self.labels_transform = labels_transform
        self.flip = flip

    def __getitem__(self, index):

        img = self.data['rgb'][index, :]
        label_org = self.data['seg'][index, :]

        if self.flip == 0:
            label_org = cv2.flip(label_org, 0)
        elif self.flip == 1:
            label_org = cv2.flip(label_org, 1)
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.labels_transform is not None:
            label = self.labels_transform(label_org)

        return img, label

    def __len__(self):
        return len(self.data['rgb'][:])
