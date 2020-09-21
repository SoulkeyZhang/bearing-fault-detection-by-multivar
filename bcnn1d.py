import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OUT_CHANNEL = 16
KERNEL_SIZE = 4
OUT_CHANNEL2 = 12
KERNEL_SIZE2 = 4
CLASS_NUM = 4
TEMP_HIDDEN = 8


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class BRCNN1D(nn.Module):
    def __init__(self, i, h, o1, o2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=i, out_channels=OUT_CHANNEL,
                      kernel_size=KERNEL_SIZE, stride=1),
            nn.BatchNorm1d(OUT_CHANNEL),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1),

            nn.Conv1d(in_channels=OUT_CHANNEL, out_channels=OUT_CHANNEL2,
                      kernel_size=KERNEL_SIZE2, stride=1),
            nn.BatchNorm1d(OUT_CHANNEL2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            Flatten(),
            nn.Dropout(0.2)
        )
        self._fc0 = nn.Linear(72, h)
        self._fc1t = nn.Linear(h, TEMP_HIDDEN)  # 训练故障类型的全连接层，比如滚珠、内外圈故障
        self._fc2t = nn.Linear(h, TEMP_HIDDEN)  # 训练故障大小的全连接层，比如0，0.07等
        self._fc1 = nn.Linear(TEMP_HIDDEN, o1)  # 训练故障类型的全连接层，比如滚珠、内外圈故障
        self._fc2 = nn.Linear(TEMP_HIDDEN, o2)  # 训练故障大小的全连接层，比如0，0.07等
        self.o1 = o1
        self.o2 = o2
        self._hidden = h

    def forward(self, x):
        cnn_output = self.cnn(x)

        # self.affine = nn.Sequential(
        #     nn.Linear(cnn_output.shape[1], self._hidden),
        # ).to(device)

        fc0 = self._fc0(cnn_output)
        fc0 = nn.functional.relu(fc0)
        fc1 = self._fc1t(fc0)
        fc1 = nn.functional.relu(fc1)
        fc1 = self._fc1(fc1)

        fc2 = self._fc2t(fc0)
        fc2 = nn.functional.relu(fc2)
        fc2 = self._fc2(fc2)

        return fc1, fc2


if __name__ == '__main__':
    from datetime import datetime
    from train import gen_seq_data, train
    import pandas as pd
    from sklearn import preprocessing  # 对数据进行标准化处理
    import torch.utils.data as tchdata
    from torch import optim

    reserveCols = ['dMean', 'sqrtAbsMean', 'waveFactor', 'maxFactor', 'dSkew', 'dKurtosis', 'dMm',  # 时域特征
                   'VF', 'meanFreq', 'DFreq', 'thirdOrderFreq', 'FC', 'sqrtMeanFreq', 'p6DivP5']  # 频域特征
#
#  将csv文件中的数据读取到dataFrame中
#
    direction = '1730_007/'
    file = 'wSize40stride40X100_DE_timebearing_data.csv'
    df_normal = pd.read_csv(direction+file)

    file = 'wSize40stride40X108_DE_timebearing_data.csv'
    df_inner = pd.read_csv(direction+file)

    file = 'wSize40stride40X121_DE_timebearing_data.csv'
    df_ball = pd.read_csv(direction+file)

    file = 'wSize40stride40X133_DE_timebearing_data.csv'
    df_outer = pd.read_csv(direction+file)

    direction = '1730_014/'

    file = 'wSize40stride40X172_DE_timebearing_data.csv'
    df_inner_14 = pd.read_csv(direction+file)
    file = 'wSize40stride40X188_DE_timebearing_data.csv'
    df_ball_14 = pd.read_csv(direction+file)
    file = 'wSize40stride40X200_DE_timebearing_data.csv'
    df_outer_14 = pd.read_csv(direction+file)

    direction = '1730_021/'

    file = 'wSize40stride40X212_DE_timebearing_data.csv'
    df_inner_21 = pd.read_csv(direction+file)
    file = 'wSize40stride40X225_DE_timebearing_data.csv'
    df_ball_21 = pd.read_csv(direction+file)
    file = 'wSize40stride40X237_DE_timebearing_data.csv'
    df_outer_21 = pd.read_csv(direction+file)
# 将特征数据按一定长度进行组装成样本，再将所有样本拼接到一起
#
#
#
    sample_nor = gen_seq_data(df_normal[reserveCols], 20)
    label_nor = np.zeros((sample_nor.shape[0], 2), dtype='int')

    sample_outer = gen_seq_data(df_outer[reserveCols], 20)
    label_outer = np.array([1, 1])
    sample_inner = gen_seq_data(df_inner[reserveCols], 20)
    label_inner = np.array([1, 2])
    sample_ball = gen_seq_data(df_ball[reserveCols], 20)
    label_ball = np.ones((sample_ball.shape[0], 2), dtype='int')
    label_ball[:, 0] = 1
    label_ball[:, 1] = 3
    label_inner = np.ones((sample_inner.shape[0], 2), dtype='int')
    label_inner[:, 0] = 1
    label_inner[:, 1] = 2
    label_outer = np.ones((sample_outer.shape[0], 2), dtype='int')
    label_outer[:, 0] = 1
    label_outer[:, 1] = 1

    sample_outer_14 = gen_seq_data(df_outer_14[reserveCols], 20)
    sample_inner_14 = gen_seq_data(df_inner_14[reserveCols], 20)
    sample_ball_14 = gen_seq_data(df_ball_14[reserveCols], 20)
    label_ball_14 = np.ones((sample_ball_14.shape[0], 2), dtype='int')
    label_ball_14[:, 0] = 2
    label_ball_14[:, 1] = 3
    label_inner_14 = np.ones((sample_inner_14.shape[0], 2), dtype='int')
    label_inner_14[:, 0] = 2
    label_inner_14[:, 1] = 2
    label_outer_14 = np.ones((sample_outer_14.shape[0], 2), dtype='int')
    label_outer_14[:, 0] = 2
    label_outer_14[:, 1] = 1

    sample_outer_21 = gen_seq_data(df_outer_21[reserveCols], 20)
    sample_inner_21 = gen_seq_data(df_inner_21[reserveCols], 20)
    sample_ball_21 = gen_seq_data(df_ball_21[reserveCols], 20)

    label_ball_21 = np.ones((sample_ball_21.shape[0], 2), dtype='int')
    label_ball_21[:, 0] = 3
    label_ball_21[:, 1] = 3
    label_inner_21 = np.ones((sample_inner_21.shape[0], 2), dtype='int')
    label_inner_21[:, 0] = 3
    label_inner_21[:, 1] = 2
    label_outer_21 = np.ones((sample_outer_21.shape[0], 2), dtype='int')
    label_outer_21[:, 0] = 3
    label_outer_21[:, 1] = 1

    samples = [sample_ball, sample_ball_14, sample_ball_21,
               sample_inner, sample_inner_14, sample_inner_21, sample_outer, sample_outer_14, sample_outer_21]
    labels = [label_ball, label_ball_14, label_ball_21, label_inner,
              label_inner_14, label_inner_21, label_outer, label_outer_14, label_outer_21]
    result = sample_nor
    rlabel = label_nor
    for i, sample in enumerate(samples):
        result = np.concatenate((result, sample))
        rlabel = np.concatenate((rlabel, labels[i]))
# 组建dataset
#
    scaler_data = result.reshape((result.shape[0], -1))
    scaler = preprocessing.StandardScaler().fit(scaler_data)
    # 将训练数据集进行标准化。此处使用该操作有隐患
    train_data = scaler.transform(scaler_data)
    train_data = train_data.reshape(result.shape)

    train_set = tchdata.TensorDataset(
        torch.from_numpy(train_data).float(), torch.from_numpy(rlabel))

    train_loader = tchdata.DataLoader(train_set, batch_size=32, shuffle=True)

# 准备训练模型
    model = BRCNN1D(14, 5, 4, 4).cuda()
    torch.backends.cudnn.benchmark = True
    optimizer1 = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.005)

    for i in range(5):
        acc = train(model, optimizer1, train_loader)
        print(datetime.now(), '\tepoch = ', i, '\ttrain acc: ', acc)
