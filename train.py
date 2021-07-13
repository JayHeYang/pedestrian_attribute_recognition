import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from tqdm._tqdm import trange
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from torch.utils.tensorboard import SummaryWriter


from utils.tools import compute_attributes_weights, accuracy, accurate_func
from utils.config import Config
from utils.my_loss import My_Loss
from models.my_net import My_Net
from data.dataset import PETADataset


def adjust_learning_rate(optimizer, epoch, decay_epoch):
    opt = Config()

    lr = opt.lr
    for epc in decay_epoch:
        if epoch >= epc:
            lr = lr * 0.1
        else:
            break
    print()
    print('Learning Rate:', lr)
    print()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train_func(times=0, seed=None):

    writer = SummaryWriter('./log' + str(times))

    opt = Config()
    net = My_Net(attribute_num=35)
    net.train() # 设置为训练模型

    train_data = PETADataset(train=True, seed=seed)
    train_dataloader = DataLoader(train_data, opt.bs, shuffle=True, num_workers=opt.nw)

    # attr_name, weights = compute_attributes_weights()
    # criterion = nn.BCEWithLogitsLoss(weight=weights)  # 查全率过低，增加阳性样本的损失，提高查全率
    criterion = My_Loss() # 查全率过低，增加阳性样本的损失，提高查全率
    optimizer = t.optim.AdamW(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.1)
    # optimizer = t.optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)

    net.cuda()
    criterion.cuda()

    if opt.pre_train:
        all_data = t.load('mtl_net_20.pth')
        net.load_state_dict(all_data['model'])
        optimizer.load_state_dict(all_data['optimizer'])

    step = 0
    for epoch in trange(opt.max_epoch):
        adjust_learning_rate(optimizer, epoch, opt.decay_epoch)

        for ii, (data, label) in enumerate(train_dataloader):
            input = V(data).cuda()
            target = V(label).cuda()

            step += 1
            optimizer.zero_grad(set_to_none=True)
            score = net(input)

            loss = criterion(score, target)

            writer.add_scalar('Loss', loss.data, step)
            loss.backward()
            optimizer.step()

            acc = accuracy(score.data, target)
            writer.add_scalar('Acc', acc, step)

            Precision, Recall, F1, Acc, mA = accurate_func(score.cpu().data, target.cpu().numpy(), sigmoid=True)

            writer.add_scalar('Train/ZhiBiao/Acc', Acc , step)
            writer.add_scalar('Train/ZhiBiao/Precision', Precision, step)
            writer.add_scalar('Train/ZhiBiao/Recall', Recall, step)
            writer.add_scalar('Train/ZhiBiao/F1', F1, step)
            writer.add_scalar('Train/ZhiBiao/mA', mA, step)

        print("\nEpoch:{}, Accuracy:{}".format(epoch + 1, acc))
        print("Epoch:{}, Loss:{}".format(epoch+1, loss.data))


        if (epoch + 1) % 1 == 0:
            all_data = dict(
                optimizer=optimizer.state_dict(),
                model=net.state_dict(),
                info=u'模型和优化器的所有参数'
            )
            t.save(all_data, 'Times_{}_mtl_net_{}.pth'.format(times, epoch+1))


if __name__ == '__main__':
    Seed = [20, 17, 21, 28, 9]
    for i in range(4, 5):
        train_func(times=i, seed=Seed[i])