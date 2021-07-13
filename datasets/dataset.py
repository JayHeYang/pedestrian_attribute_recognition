import os
from PIL import Image
from torch.utils import data
import random
import numpy as np
from torchvision import transforms as T
import pickle


def getRandomIndex(n, x, seed):
	# 索引范围为[0, n), 随机选x个不重复
    random.seed(seed) # 设置随机数种子
    index = random.sample(range(n), x)
    return index



class PETADataset(data.Dataset):

    def __init__(self, root='peta_release/images',
                 transforms=None, train=True, test=False, seed=None):
        self.test = test

        # path = 'peta_release/my_peta.pkl'

        path = '/Users/morvan/Downloads/peta_release/my_peta.pkl'
        f = open(path, 'rb')
        data = pickle.load(f)

        imgs = [root + '/' + f'{i + 1:05}.png' for i in range(19000)]

        self.labels = data['label']

        imgs_num = len(imgs)

        test_num = int(imgs_num * 0.4)

        # 先根据上面的函数获取test_index
        test_index = np.array(getRandomIndex(imgs_num, test_num, seed))
        # 再将test_index从总的index中减去就得到了train_index
        train_index = np.delete(np.arange(imgs_num), test_index)


        if train:
            self.imgs = list(np.array(imgs)[train_index])
        else:
            self.imgs = list(np.array(imgs)[test_index])


        if transforms is None:

            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            if train:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.Resize([187, 119]),
                    T.RandomCrop((187, 119), 10),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize([187, 119]),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label_index = int(img_path.split('/')[-1].split('.')[0]) - 1
        label = np.array(self.labels[label_index], dtype='float32')
        image = Image.open(img_path)
        image = self.transforms(image)
        return image, label



    def __len__(self):
        return len(self.imgs)