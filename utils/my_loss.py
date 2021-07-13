import torch.nn as nn
import torch as t
import torch
import torch.nn.functional as func

class My_Loss(nn.Module):
    def __init__(self):
        super(My_Loss, self).__init__()


        self.weights = torch.Tensor([0.5016,
                                        0.3275,
                                        0.1023,
                                        0.0597,
                                        0.1986,
                                        0.2011,
                                        0.8643,
                                        0.8559,
                                        0.1342,
                                        0.1297,
                                        0.1014,
                                        0.0685,
                                        0.314,
                                        0.2932,
                                        0.04,
                                        0.2346,
                                        0.5473,
                                        0.2974,
                                        0.0849,
                                        0.7523,
                                        0.2717,
                                        0.0282,
                                        0.0749,
                                        0.0191,
                                        0.3633,
                                        0.0359,
                                        0.1425,
                                        0.0454,
                                        0.2201,
                                        0.0178,
                                        0.0285,
                                        0.5125,
                                        0.0838,
                                        0.4605,
                                        0.0124]).cuda()
        self.EPS = 1e-12


    def forward(self, Score, Target):

        # self.weights = weights
        EPS = self.EPS
        Score = t.sigmoid(Score)

        cur_weights = torch.exp(Target + (1 - Target * 2) * self.weights)
        loss = cur_weights * (Target * torch.log(Score + EPS)) + ((1 - Target) * torch.log(1 - Score + EPS))

        return torch.neg(torch.mean(loss))




# if __name__ == '__main__':
#
#     w = t.ones(35)
#
#     criterion1 = My_Loss()
#     criterion2 = nn.BCEWithLogitsLoss()
#
#     score = t.randn(10, 35)
#     target = t.ones(10, 35)
#
#     # 随机设置一些索引为0
#     index = t.randn(10, 35) < 1
#     target[index] = 0
#
#     print(criterion1(score.cuda(), target.cuda()))
#     print(criterion2(score, target))
