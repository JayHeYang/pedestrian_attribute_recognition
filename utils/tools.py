"""
一些工函数
"""
import numpy as np
import torch as t
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve
import pickle



def compute_attributes_weights():
    """
    计算35个标签属性的权值
    :return:
    """
    path = '/home/mist/peta_release/my_peta.pkl'
    f = open(path, 'rb')
    data = pickle.load(f)
    labels = data['label']
    attr_name = data['attr_name']

    counts = np.sum(labels, axis=0)
    fre = counts / 19000


    weights = np.exp(- fre / 0.8)

    weights = t.Tensor(weights)


    return attr_name, weights


def compute_batch_attributes_weights(Target):

    counts = t.sum(Target, dim=0)

    N = Target.size()[0] # batchsize 的大小
    zero_idx = counts == 0
    counts[zero_idx] = 1

    weights = counts / N

    return weights

def accurate_func(score, target, sigmoid=False):
    '''
    计算相关指标
    :param score: 预测得分(未经过sigmoid) 10000*35
    :param target: 标签值 10000*35
    :param sigmoid: 是否需要进行sigmoid处理
    :return: p, r, f1, acc
    '''
    if sigmoid:
        score = t.sigmoid(t.Tensor(score))

    # N = score.size()[0]

    # 处理预测结果，大于0.5则视为1, 小于则视为0
    mask0 = score.lt(0.5)
    mask1 = score.ge(0.5)
    score[mask0] = 0
    score[mask1] = 1

    ## numpy 化, target本来就是数组，不用numpy（）
    score = score.numpy()

    TP = np.sum((score == 1) & (target == 1), axis=1)
    TN = np.sum((score == 0) & (target == 0), axis=1)
    FN = np.sum((score == 0) & (target == 1), axis=1)
    FP = np.sum((score == 1) & (target == 0), axis=1)


    p = np.mean(TP / (TP + FP))
    r = np.mean(TP / (TP + FN))
    f1 = 2 * r * p / (r + p)
    acc = np.mean(TP / (TP + FP + FN))


    pos_cnt = np.sum((score == 1) & (target == 1), axis=0)
    pos_tol = np.sum(target == 1, axis=0)
    neg_cnt = np.sum((score == 0) & (target == 0), axis=0)
    neg_tol = np.sum(target == 0, axis=0)

    ma = np.mean((pos_cnt / pos_tol + neg_cnt / neg_tol) / 2)

    return p, r, f1, acc, ma





def ROC_curve(score, target, sigmoid=False):

    if sigmoid:
        score = t.sigmoid(t.Tensor(score)).numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(35):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


    return roc_auc["micro"]



def PR_curve(score, target, sigmoid=False):
    if sigmoid:
        score = t.sigmoid(t.Tensor(score)).numpy()

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(35):
        precision[i], recall[i], _ = precision_recall_curve(target[:, i],
                                                            score[:, i])
        average_precision[i] = average_precision_score(target[:, i], score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(target.ravel(),
                                                                    score.ravel())
    average_precision["micro"] = average_precision_score(target, score,
                                                         average="micro")


    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))

    return average_precision["micro"]



def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = torch.sigmoid(output).cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()

    res = []
    for k in range(attr_num):
        res.append(1.0*sum(correct[:,k]) / batch_size)
    return sum(res) / attr_num






# if __name__ == '__main__':

    # score = np.random.randn(10, 8)
    # a = [0, 1, 1, 0, 0, 1, 0, 1]
    # b = [1, 1, 1, 0, 1, 0, 1, 1]
    # target = np.array([a, b, b, a, a, b, a, b, a, b]).reshape(10, 8)
    #
    # print(PR_curve(score, target, sigmoid=True))
    # compute_attributes_weights()
