import torch
import torch.utils.data as data

import numpy as np

class PROMISE12(data.Dataset):
    def __init__(self, mode, images, GT=None, transform=None, GT_transform=None):
        if images is None:
            raise(RuntimeError("images must be set"))
        self.images = images
        self.GT = GT
        self.transform = transform
        self.GT_transform = GT_transform

    def __getitem__(self, index):
        """
        Args:
            index(int): Index
        Returns:
            tuple: (image, GT) where GT is index of the
        """
        if self.mode == "train" or self.mode == "test":
            keys = list(self.images.keys())
            id = keys[index]
            image = self.images[id]
            print("image shape from DataManager shown in PROMISE12:" + str(image.shape))
            x, y, z = image.shape # added by Chao
            image = image.reshape((1, z, y, x)) # added by Chao
            image = image.astype(np.float32)
            if self.transform is not None:
                print(type(image))
                image = torch.from_numpy(image)
                # image = self.transform(image)

            print(id + "\n")
            if self.GT is None:
                return image, id
            else:
                GT = self.GT[id[:-4] + '_segmentation' + '.mhd']
                if self.GT_transform is not None:
                    GT = self.GT_transform(GT)
                return image, GT, id
        elif self.mode == "infer":
            keys = list(self.images.keys())
            id = keys[index]
            image = self.images[id]
            print("image shape from DataManager shown in PROMISE12:" + str(image.shape))
            # no image reshape needed for "infer" compared to "train" or "test".
            image = image.astype(np.float32)
            return image, id

    def __len__(self):
        return len(self.images)