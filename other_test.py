
import torch as t
import torch.nn as nn
import numpy as np
import torch
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


biaopqian =  ['Age16-30',
            'Age31-45',
            'Age46-60',
            'AgeAbove61',
            'Backpack',
            'CarryingOther',
            'Casual lower',
            'Casual upper',
            'Formal lower',
            'Formal upper',
            'Hat',
            'Jacket',
            'Jeans',
            'Leather Shoes',
            'Logo',
            'Long hair',
            'Male',
            'Messenger Bag',
            'Muffler',
            'No accessory',
            'No carrying',
            'Plaid',
            'PlasticBags',
            'Sandals',
            'Shoes',
            'Shorts',
            'Short Sleeve',
            'Skirt',
            'Sneaker',
            'Stripes',
            'Sunglasses',
            'Trousers',
            'Tshirt',
            'UpperOther',
            'V-Neck']



def test(e, description, times=0, seed=666):


    path = 'Times_{}_mtl_net_{}.pth'.format(times, e)
    # 记录每个epoch产生的指标信息

    t.no_grad()
    opt = Config()
    model = My_Net(attribute_num=opt.attr_num)
    attr_num = opt.attr_num
    # 加载已经训练好的模型
    all_data = t.load(path)
    model.load_state_dict(all_data['model'])
    model.cuda()

    test_data = PETADataset(train=False, seed=seed)
    val_loader = DataLoader(test_data, batch_size=opt.bs, shuffle=True,
                                 num_workers=opt.nw)
    model.eval()

    pos_cnt = []
    pos_tol = []
    neg_cnt = []
    neg_tol = []

    accu = 0.0
    prec = 0.0
    recall = 0.0
    tol = 0

    for it in range(attr_num):
        pos_cnt.append(0)
        pos_tol.append(0)
        neg_cnt.append(0)
        neg_tol.append(0)

    for i, _ in enumerate(val_loader):
        input, target = _
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        output = model(input)
        bs = target.size(0)

        # # maximum voting
        # if type(output) == type(()) or type(output) == type([]):
        #     output = torch.max(torch.max(torch.max(output[0],output[1]),output[2]),output[3])


        batch_size = target.size(0)
        tol = tol + batch_size
        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        target = target.cpu().numpy()

        for it in range(attr_num):
            for jt in range(batch_size):
                if target[jt][it] == 1:
                    pos_tol[it] = pos_tol[it] + 1
                    if output[jt][it] == 1:
                        pos_cnt[it] = pos_cnt[it] + 1
                if target[jt][it] == 0:
                    neg_tol[it] = neg_tol[it] + 1
                    if output[jt][it] == 0:
                        neg_cnt[it] = neg_cnt[it] + 1

        if attr_num == 1:
            continue
        for jt in range(batch_size):
            tp = 0
            fn = 0
            fp = 0
            for it in range(attr_num):
                if output[jt][it] == 1 and target[jt][it] == 1:
                    tp = tp + 1
                elif output[jt][it] == 0 and target[jt][it] == 1:
                    fn = fn + 1
                elif output[jt][it] == 1 and target[jt][it] == 0:
                    fp = fp + 1
            if tp + fn + fp != 0:
                accu = accu +  1.0 * tp / (tp + fn + fp)
            if tp + fp != 0:
                prec = prec + 1.0 * tp / (tp + fp)
            if tp + fn != 0:
                recall = recall + 1.0 * tp / (tp + fn)
    print('Epoch=',e)
    print('=' * 100)
    print('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
    mA = 0.0
    for it in range(attr_num):
        cur_mA = ((1.0*pos_cnt[it]/pos_tol[it]) + (1.0*neg_cnt[it]/neg_tol[it])) / 2.0
        mA = mA + cur_mA
        print('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,description[it],pos_cnt[it],neg_cnt[it],pos_tol[it],neg_tol[it],(pos_cnt[it]+neg_tol[it]-neg_cnt[it]),(neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA))
    mA = mA / attr_num
    print('\t' + 'mA:        '+str(mA))

    if attr_num != 1:
        accu = accu / tol
        prec = prec / tol
        recall = recall / tol
        f1 = 2.0 * prec * recall / (prec + recall)
        print('\t' + 'Accuracy:  '+str(accu))
        print('\t' + 'Precision: '+str(prec))
        print('\t' + 'Recall:    '+str(recall))
        print('\t' + 'F1_Score:  '+str(f1))

        writer.add_scalar('Test/ZhiBiao/Acc', accu , e)
        writer.add_scalar('Test/ZhiBiao/Precision', prec, e)
        writer.add_scalar('Test/ZhiBiao/Recall', recall, e)
        writer.add_scalar('Test/ZhiBiao/F1', f1, e)
        writer.add_scalar('Test/ZhiBiao/mA', mA, e)
    print('=' * 100)


if __name__ == '__main__':
    Seed = [20, 17, 21, 28, 9]
    for i in trange(4, 5):
        writer = SummaryWriter('./log' + str(i))
        for j in trange(1, 61):
            test(j, biaopqian, times=i, seed=Seed[i])