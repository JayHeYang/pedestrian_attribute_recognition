class Config(object):

    nw = 4 # 多线程加载数据所用的线程数
    # bs = 64 # batch_size
    bs = 32  # batch_size
    lr = 0.001 # learning rate
    max_epoch = 60 # max epoch

    wd = 0.005 # weight decay 权重惩罚正则化项
    momentum = 0.9 # 动量，防止梯度方向180°转变
    attr_num = 35 # 属性标签数量
    decay_epoch = [20, 40]

    pre_train = False # 是否加载训练好的模型
    pre_train_path = 'Times_4_mtl_net_60.pth'
    # use_gpu = False # 使用GPU进行训练、预测

    
