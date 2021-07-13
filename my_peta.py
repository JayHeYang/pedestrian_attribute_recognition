import os
import numpy as np
import random

import pickle
from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(666)
random.seed(666)

# 打乱属性顺序
# group_order = np.random.permutation(np.arange(35))


# 创建存储目录pip
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


# 处理初始.mat文件——筛选需要的属性，然后利用pickle打包
def generate_data_description(save_dir, reorder):

    peta_data = loadmat(os.path.join(save_dir, 'PETA.mat'))
    dataset = EasyDict()

    dataset.description = 'peta dataset with cleaned'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'images')
    dataset.image_name = [f'{i+1:05}.png' for i in range(19000)]

    # 所有的属性名称
    all_attr_name = [i[0][0] for i in peta_data['peta'][0][0][1]]

    # 所有图片对应的标签,前四个信息都不重要——>(19000, 105)
    all_label = peta_data['peta'][0][0][0][:, 4:]

    dataset.attr_name = all_attr_name[:35]
    # shape——>(19000, 10)
    dataset.label = all_label[:, :35]

    # 打乱属性和标签顺序
    # if reorder:
    #     dataset.label = dataset.label[:, group_order]
    #     dataset.attr_name = [dataset.attr_name[i] for i in group_order]


    dataset.partition = EasyDict()
    dataset.partition.train = []
    dataset.partition.val = []
    dataset.partition.test = []
    dataset.partition.trainval = []


    # 划分训练、验证、测试集，分为五份

    for idx in range(5):
        train = peta_data['peta'][0][0][3][idx][0][0][0][0][:, 0] - 1
        val = peta_data['peta'][0][0][3][idx][0][0][0][1][:, 0] - 1
        test = peta_data['peta'][0][0][3][idx][0][0][0][2][:, 0] - 1
        trainval = np.concatenate((train, val), axis=0)

        dataset.partition.train.append(train)
        dataset.partition.val.append(val)
        dataset.partition.test.append(test)
        dataset.partition.trainval.append(trainval)

        weight_train = np.mean(dataset.label[train], axis=0)
        weight_trainval = np.mean(dataset.label[trainval], axis=0)

        dataset.weight_train = weight_train
        dataset.weight_trainval = weight_trainval

    # 将所有数据属性名、照片路径、标签、数据集划分、初始权重等打包存入pickle文件
    with open(os.path.join(save_dir, 'my_peta.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    save_dir = 'peta_release'

    generate_data_description(save_dir, reorder=False)