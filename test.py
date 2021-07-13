import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


from tqdm._tqdm import trange
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from torch.utils.tensorboard import SummaryWriter


from utils.tools import compute_attributes_weights, accurate_func, ROC_curve, PR_curve
from utils.config import Config
from models.my_net import My_Net
from data.dataset import PETADataset


writer = SummaryWriter('./log')

def test(a = 1):

    path = 'mtl_net_' + str(a) + '.pth'
    # 记录每个epoch产生的指标信息

    t.no_grad()
    opt = Config()
    model = My_Net(attribute_num=opt.attr_num)

    # 加载已经训练好的模型
    all_data = t.load(path)
    model.load_state_dict(all_data['model'])
    model.cuda()

    test_data = PETADataset(train=False)
    test_dataloader = DataLoader(test_data, batch_size=opt.bs, shuffle=True,
                                 num_workers=opt.nw)

    for ii, (data, label) in enumerate(test_dataloader):
        input = V(data)
        input = input.cuda()

        score = model(input)
        score = score.cpu().detach().numpy()

        target = label.numpy()

        if ii == 0:
            Score = score
            Target = target
        else:
            Score = np.concatenate((Score, score) ,axis=0)
            Target = np.concatenate((Target, target), axis=0)

        print('Loading.....')


    # # 保存输入输出数据
    # np.save('Score', Score)
    # np.save('Target', Target)

    Precision, Recall, F1, Acc, mA = accurate_func(Score, Target, sigmoid=True)

    # AUC = ROC_curve(Score, Target, sigmoid=True)
    # AP = PR_curve(Score, Target, sigmoid=True)

    writer.add_scalar('Test/ZhiBiao/Acc', Acc * 100, a)
    writer.add_scalar('Test/ZhiBiao/Precision', Precision * 100, a)
    writer.add_scalar('Test/ZhiBiao/Recall', Recall * 100, a)
    writer.add_scalar('Test/ZhiBiao/F1', F1 * 100, a)
    writer.add_scalar('Test/ZhiBiao/mA', mA * 100, a)

    print('Epoch: %d' % a)
    print('测试集属性的平均准确率Acc:%.2f %%' % (Acc*100))
    print('测试集属性的平均精准率为Prec:%.2f %%' % (Precision*100))
    print('测试集属性的平均召回率为Recall:%.2f %%' % (Recall*100))
    print('测试集属性的平均F1分数为F1:%.2f %%' % (F1*100))

    print('测试集属性的平均正确率为mA:%.2f %%' % (mA * 100))
    # print('测试集ROC曲线的AUC分数为AUC:%.2f %%' % (AUC * 100))
    # print('测试集PR曲线的AP分数为AP:%.2f %%' % (AP * 100))

if __name__ == '__main__':
    for i in range(1, 61):
        test(a=i)
