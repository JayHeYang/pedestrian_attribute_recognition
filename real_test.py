import torch as t
import numpy as np
import cv2
import random
import pickle
from PIL import Image

from torch.autograd import Variable as V
from torchvision import transforms as T


from utils.config import Config
from models.my_net import My_Net





biaopqian =  ['Age16-30',
            'Age31-45',
            'Age46-60',
            'AgeAbove61',
            'Backpack',
            'CarryingOther', # 携带某些东西
            'Casual lower',
            'Casual upper',
            'Formal lower',
            'Formal upper',
            'Hat',
            'Jacket', # 夹克
            'Jeans', # 牛仔裤qq
            'Leather Shoes', # 皮鞋
            'Logo',
            'Long hair',
            'Male',
            'Messenger Bag', # 斜挎包
            'Muffler', # 围巾
            'No accessory', # 没有装饰物
            'No carrying', # 没有携带东西
            'Plaid',  # 上身格子纹
            'PlasticBags', # 携带塑料袋
            'Sandals', # 拖鞋，凉鞋
            'Shoes',  # 鞋
            'Shorts', # 短裤
            'Short Sleeve', # 短袖
            'Skirt', # 裙子
            'Sneaker', # 运动鞋
            'Stripes', # 下身条纹
            'Sunglasses', # 太阳镜
            'Trousers', # 裤子
            'Tshirt',
            'UpperOther', # 上身另外的穿搭
            'V-Neck'] # V领

biaopqian = np.array(biaopqian)


path = 'Times_4_mtl_net_60.pth'

t.no_grad()
opt = Config()
model = My_Net(attribute_num=opt.attr_num)

# 加载已经训练好的模型
all_data = t.load(path, map_location='cpu')
model.load_state_dict(all_data['model'])
model.eval()


transforms = T.Compose([
                    T.Resize([187, 119]),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
])



# OpenCV调用摄像头获取图像检测
cap = cv2.VideoCapture(0)
#

# cap = cv2.VideoCapture('/Users/morvan/Movies/IMG_2136.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_camera_real.mp4',fourcc, 20.0, (1000+240*2, 600))


while(True):
    ret, frame = cap.read()

    frame1 = frame[...,::-1]

    # 更改图片尺寸
    frame = cv2.resize(frame, (1000, 600))

    frame1 = cv2.resize(frame1, (1000, 600))

    img = Image.fromarray(frame1)
    # 获取图片尺寸
    height = np.shape(img)[0]
    width = np.shape(img)[1]

    #  生成虚拟的图片
    # fake_img = np.zeros((height, 240, 3), np.uint8)
    fake_img = np.zeros((600, 240, 3), np.uint8)

    img = transforms(img=img).view(-1, 3, 187, 119)
    input = V(img)

    score = model(input)
    score = t.sigmoid(t.Tensor(score.cpu()))

    mask0 = score.lt(0.5)
    mask1 = score.ge(0.5)
    score[mask0] = 0
    score[mask1] = 1

    print("Predict Label: ", score)
    print('\n')

    pred_biaoqian = biaopqian[score.bool().numpy().flatten()]

    print('预测标签：', pred_biaoqian)
    print('\n')

    # 构造新的imgs 加上两边黑框
    frame = cv2.hconcat([fake_img, frame, fake_img])
    for ii, text in enumerate(pred_biaoqian):
        cv2.putText(frame, text, (0, 30+ii*30), font, 1, (238, 99, 99), 2, cv2.LINE_AA)

    cv2.imshow('PAR', frame)
    out.write(frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break

# 释放摄像头，关闭显示窗口
cap.release()
out.release()
cv2.destroyAllWindows()



# #
# ## PETA数据集图像测试
#
# f = open('/Users/morvan/Downloads/peta_release/my_peta.pkl', 'rb')
# data = pickle.load(f)
#
# # 真实标签
# labels = data['label']
#
#
# # suiji_biaohao = [511, 614, 7028, 335, 10752, 10719, 7082, 7447, 122, 7228, 7122, 17202]
# random.seed(666)
# suiji_biaohao = random.sample(range(19000), 100)
#
# Acc = []
# Prec = []
# Recall = []
# for pic_index in suiji_biaohao:
#
#     path = '/Users/morvan/Downloads/peta_release/images/' + f'{pic_index:05}.png'
#     print(path)
#
#     img1 = cv2.imread(path)
#     imgs = Image.open(path)
#     # 获取图片尺寸
#     width = np.shape(imgs)[0]
#     height = np.shape(imgs)[1]
#
#
#
#     #  生成虚拟的图片
#     fake_img = np.zeros((width,100,3),np.uint8)
#     print(np.shape(fake_img))
#
#     # img = Image.fromarray(imgs)
#     img = transforms(img=imgs).unsqueeze(0)
#     input = V(img)
#
#
#     score = model(input)
#     score = t.sigmoid(t.Tensor(score.cpu()))
#
#     mask0 = score.lt(0.5)
#     mask1 = score.ge(0.5)
#     score[mask0] = 0
#     score[mask1] = 1
#
#     score = score.bool().numpy().flatten()
#     tp = 0
#     fn = 0
#     fp = 0
#
#
#     # 计算指标
#     for k in range(35):
#         if score[k] == 1 and labels[pic_index-1][k] == 1:
#             tp += 1
#         elif score[k] == 0 and labels[pic_index-1][k] == 1:
#             fn += 1
#         elif score[k] == 1 and labels[pic_index-1][k] == 0:
#             fp += 1
#     acc = tp / (tp + fn + fp)
#     prec =  tp / (tp + fp)
#     recall = tp / (tp + fn)
#     Acc.append(acc)
#     Prec.append(prec)
#     Recall.append(recall)
#
#     # print("acc:{}, prec:{}, recall:{}".format(acc, prec, recall))
#
#     if acc > 0.8 and prec > 0.8 and recall > 0.8:
#
#         pred_biaoqian = biaopqian[score]
#         print('预测标签：', pred_biaoqian)
#         print('\n')
#
#         true_labels =  biaopqian[np.array(labels[pic_index], dtype=bool)]
#         print('真实标签：', true_labels)
#         print('\n')
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#
#
#         # 构造新的imgs 加上两边黑框
#         imgs = cv2.hconcat([fake_img, img1, fake_img])
#
#         for ii, text in enumerate(pred_biaoqian):
#             # 添加预测标签 左侧
#             cv2.putText(imgs, text, (0, 10+ii*10), font, 0.3, (65,105,225), 1, cv2.LINE_AA)
#
#         cv2.putText(imgs, 'Pred_Label', (20, width-10), font, 0.3, (65,105,225), 1, cv2.LINE_AA)
#
#
#         for ii, text in enumerate(true_labels):
#             # 添加真实标签 右侧
#             cv2.putText(imgs, text, (100 + height, 10 + ii * 10), font, 0.3, (0, 128, 0), 1, cv2.LINE_AA)
#
#         cv2.putText(imgs, 'True_Label', (100 + height + 20, width - 10), font, 0.3, (0, 128, 0), 1, cv2.LINE_AA)
#         # cv2.imshow('frame', imgs)
#
#         cv2.imwrite("peta_result_{}.jpg".format(pic_index),imgs)
#     else:
#         pass
#
# print(print("acc:{}, prec:{}, recall:{}".format(np.mean(Acc), np.mean(Prec), np.mean(Recall))))
#



# 单张测试
# for idx in range(1, 59):
#     path = '/Users/morvan/Desktop/视频素材/照片素材/{}.jpg'.format(idx)
#     print(path)
#
#     imgs1 = cv2.imread(path)
#     imgs = Image.open(path)
#     # 获取图片尺寸
#     width = np.shape(imgs)[0]
#     height = np.shape(imgs)[1]
#
#     #  生成虚拟的图片
#     fixed_h = 150
#     fake_img = np.zeros((width, fixed_h, 3), np.uint8)
#     print(np.shape(fake_img))
#
#     # img = Image.fromarray(imgs)
#     img = transforms(img=imgs).view(1, 3, 187, 119)
#     input = V(img)
#
#
#     score = model(input)
#     score = t.sigmoid(t.Tensor(score.cpu()))
#
#     mask0 = score.lt(0.5)
#     mask1 = score.ge(0.5)
#     score[mask0] = 0
#     score[mask1] = 1
#
#     # print("Predict Label: ", score)
#     # print('\n')
#
#     pred_biaoqian = biaopqian[score.bool().numpy().flatten()]
#     print('预测标签：', pred_biaoqian)
#     print('\n')
#
#
#     font = cv2.FONT_HERSHEY_SIMPLEX
#
#     # 构造新的imgs 加上两边黑框
#     imgs = cv2.hconcat([fake_img, imgs1, fake_img])
#
#     for ii, text in enumerate(pred_biaoqian):
#         # 添加预测标签 左侧
#         cv2.putText(imgs, text, (0, 20+ii*20), font, 0.5, (65,105,225), 1, cv2.LINE_AA)
#
#     cv2.putText(imgs, 'Pred_Label', (20, width-10), font, 0.5, (65,105,225), 1, cv2.LINE_AA)
#     cv2.imwrite("new_result_{}.jpg".format(idx),imgs)


