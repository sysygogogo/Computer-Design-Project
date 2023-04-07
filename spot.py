from torch.nn import init
# from train_rgb_PyTorch import PyTorch_Net
import torch
from skimage import io
import torch.nn as nn
import pprint
import artdaq
from artdaq.constants import AcquisitionType, TerminalConfiguration
import numpy as np
import pandas as pd
import os
from PIL import Image
import math

# classifier_data = os.listdir("./test_pic")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Flatten(nn.Module):
    def forward(self, input):
        input = input.contiguous()
        return input.view(input.size(0), -1)


class PyTorch_Net(nn.Module):
    def __init__(self):
        super(PyTorch_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc = nn.Linear(960, 64, bias=True)
        self.classifier = nn.Linear(64, 11, bias=True)
        self.Flatten = Flatten()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=-1)

        self.initialize_weights()

    def initialize_weights(self):
        init.normal_(self.conv1.weight.data, mean=0, std=0.1)
        init.normal_(self.conv2.weight.data, mean=0, std=0.1)
        init.normal_(self.fc.weight.data, mean=0, std=0.1)
        init.normal_(self.classifier.weight.data, mean=0, std=0.1)
        init.zeros_(self.conv1.bias.data)
        init.zeros_(self.conv2.bias.data)
        init.zeros_(self.fc.bias.data)
        init.zeros_(self.classifier.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.maxpool(x)
        x = self.Flatten(x)

        x = self.fc(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def prediect(img_path):
    folder = ['airc', 'bulb', 'fan', 'fridge', 'hairdryer', 'heater', 'lamp', 'laptop', 'micro', 'vacuum', 'washer']
    # 加载网络模型
    net = PyTorch_Net()
    model_dict = torch.load("vi_model/cnn_model.t")
    net.load_state_dict(model_dict['net'])
    net.eval()
    net = net.to(device)

    # 加载图片
    img = io.imread(img_path)
    img = np.reshape(img, (1, -1))  # 将图片像素值转换成一行  长度(1500)
    img = np.array(img).astype(float)
    img = img.reshape(-1, 20, 25, 3)
    img = torch.from_numpy(np.transpose(img, (0, 3, 1, 2)))
    img = torch.tensor(img, dtype=torch.float32)
    # x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    img = img.to(device)
    output = net(img)

    softmax = nn.Softmax(dim=-1)
    prediction = softmax(output).squeeze()
    prediction = torch.nonzero(prediction == torch.max(prediction), as_tuple=False).squeeze()
    print("该用电器的种类为：" + folder[prediction])


if __name__ == '__main__':
    # csv采集
##################################################################
    time = 10  # 采样时长/s
    sample_rate = 44100  # 采样频率
    sample_len = sample_rate * time  # 采样点数

    pp = pprint.PrettyPrinter(indent=4)

    with artdaq.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0",
                                             terminal_config=TerminalConfiguration.DIFFERENTIAL)  # 设置接口,采集模式：差分模式
        # task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config=TerminalConfiguration.DIFFERENTIAL)
        # task.ai_channels.add_ai_voltage_chan("Dev1/ai1")

        task.timing.cfg_samp_clk_timing(
            sample_rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=2)  # samps_per_chan=5 每通道读5个数据点
        print("开始采集")

        for i in range(1):
            data = task.read(number_of_samples_per_channel=sample_len,
                             timeout=artdaq.constants.WAIT_INFINITELY)  # 超时时间无限

    for i in range(len(data)):
        if i == 0:
            df_result = pd.DataFrame((np.array(data[i])) * 0.3 * 2.5)  # 电流换算
        else:
            df = pd.DataFrame((np.array(data[i])) * 420)  # 电压换算

            df_result = pd.concat([df_result, df], axis=1)
            df_result.columns = ['current', 'voltage']

    df_result.to_csv("./test1/testtest/test.csv", index=False)  # 每次采集需要修改保存文件名称，可改成绝对路径存入其他文件夹

    print("采集和存入csv已经完成！")

    # 图像转换
########################################################################################
    # Data_path = r'E:/负荷分解数据集/PLAID/'
    Data_path = './test1/'
    folders = os.listdir(Data_path)

    def youxiaodianliu(data):
        MAX_V = max(abs(data['voltage']))
        RMS_V = MAX_V / math.sqrt(2)
        p_active = 0
        for i in range(500):
            p_active = p_active + data['current'][i] * data['voltage'][i]
        p_active = p_active / 500
        # i = ia + if ;  if = i - ia ; ia = P *v(t) /(V*V)
        for i in range(500):
            # if：
            data['current'][i] = data['current'][i] - (data['voltage'][i] * p_active) / (RMS_V * RMS_V)

        # for i in range(500):
        # ia：
        #     data['current'][i] = (data['voltage'][i] * p_active) / (RMS_V * RMS_V)
        return data


    def data_process(data):  # 滤波
        dianya = data['voltage'][-5000:]
        dianliu = data['current'][-5000:]
        # 将每两个周期变成一列，合并成dataframe用于求均值
        df = pd.DataFrame(
            list(zip(dianya[0:1000], dianya[1000:2000], dianya[2000:3000], dianya[3000:4000], dianya[4000:5000])))
        # print(df)
        df_mean = df.mean(axis=1)  # 求均值
        # print(df_mean)

        dt = pd.DataFrame(
            list(zip(dianliu[0:1000], dianliu[1000:2000], dianliu[2000:3000], dianliu[3000:4000], dianliu[4000:5000])))
        # print(dt)
        dt_mean = dt.mean(axis=1)
        # print(dt_mean)

        data_save = pd.DataFrame(list(zip(dt_mean, df_mean)))
        data_save.columns = ['current', 'voltage']
        # print(data_save)

        return data_save


    def datatomatrix(data):
        for i in range(0, 500, 25):
            if i == 0:
                a = data[i:i + 25].values  # 将四行数据组成一个二维矩阵
                aa = a[None, :]  # 升维为三维矩阵(1,a,b)
                # print(aa)
                # print(aa.shape)
            else:
                arr1 = data[i:i + 25].values
                # print(arr1)
                bb = np.append(aa, arr1)  # 先拼接成一个行向量
                dim = aa.shape  # 获取原矩阵的维数
                aa = bb.reshape(dim[0] + 1, dim[1], dim[2])  # 再通过原矩阵的维数重新组合

        # print('合并矩阵：\n',aa)
        # print('维数：',aa.shape)
        return aa


    def btoP(data):  # 有功功率（固定值）
        p_active = 0
        for i in range(500):
            p_active = p_active + data['current'][i] * data['voltage'][i]
        p_active = p_active / 500

        for i in range(500):
            data['voltage'][i] = p_active
        return data


    def btoIPW(data):  # 瞬时功率波形

        for i in range(500):
            data['voltage'][i] = data['current'][i] * data['voltage'][i]
        return data


    def wuxiao(data):
        MAX_V = max(abs(data['voltage']))
        RMS_V = MAX_V / math.sqrt(2)
        p_active = 0
        for i in range(500):
            p_active = p_active + data['current'][i] * data['voltage'][i]
        p_active = p_active / 500

        for i in range(500):
            data['current'][i] = data['current'][i] - (data['voltage'][i] * p_active) / (RMS_V * RMS_V)
        return data


    # 循环
    for ind, folder in enumerate(folders):
        print(folder)
        # csv_flies = os.listdir(Data_path + folder + '/')
        csv_flies = os.listdir(Data_path + folder)
        # if folder == 'washer':
        # print(Data_path+folder+'/')
        # print(csv_flies)
        for ind, csv_flie in enumerate(csv_flies):
            # print(csv_flie)
            pic_name = csv_flie[:-4]  # 去掉字符串中的“.csv”if
            save_path = './pic - 空副本/' + folder + '/'

            # print(save_path)
            print(pic_name)  # 图片的名字
            # print(Data_path+folder+'/'+ csv_flie)
            data = pd.read_csv(Data_path + folder + '/' + csv_flie)  # , names=['current', 'voltage']

            pic = data_process(data)[0:500]  # 原数据

            pic1 = pic.copy(deep=True)
            pic2 = pic.copy(deep=True)
            pic3 = pic.copy(deep=True)

            pic1 = btoIPW(pic1)
            # print(pic1)
            #
            # x = range(0, len(pic1['current']))
            # plt.plot(x, pic1['current'])
            # plt.show()
            # x = range(0, len(pic1['voltage']))
            # plt.plot(x, pic1['voltage'])
            # plt.show()

            # pic2 = wuxiao(pic2)

            # x = range(0, len(pic['current']))
            # plt.plot(x, pic['current'])
            # plt.show()

            # x = range(0, len(pic['voltage']))
            # plt.plot(x, pic['voltage'])
            # plt.show()

            youxiaodianliu(pic2)
            # VIf轨迹代码：###########################################################################################
            data_hs = youxiaodianliu(pic2)
            data_hs = data_hs.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # 归一化后
            ########################################################################################################
            # V-I轨迹：
            # data_hs = pic3.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))  # 归一化后
            ########################################################################################################

            # print(pic)
            vitrace = data_hs.copy(deep=True)  # 深拷贝
            vitrace['current'] = vitrace['current'] * 25
            vitrace['voltage'] = vitrace['voltage'] * 20

            # vitrace.replace(np.nan, 0, inplace=True)
            # vitrace.replace(np.inf, 0, inplace=True)
            vitrace = vitrace.astype(int)
            # print(vitrace)
            VIaisle = np.zeros((25, 20))
            for n in range(500):
                if (vitrace['current'][n] == 25):  # 防止出界
                    vitrace['current'][n] = 24
                if (vitrace['voltage'][n] == 20):
                    vitrace['voltage'][n] = 19
                Horizontal = vitrace['current'][n]
                Vertical = vitrace['voltage'][n]
                VIaisle[Horizontal, Vertical] = 1

            VIaisle = np.rot90(VIaisle)  ##将矩阵img逆时针旋转90°
            # print(VIaisle)
            # VIaisle = 255 - VIaisle * 255  # 形成只有0和255的矩阵 0-黑 255-白，轨迹为黑色
            VIaisle = VIaisle * 255  # 轨迹为白色 0-黑 255-白

            # print(pic)

            VIaisle = VIaisle.reshape(-1)
            VIaisle = VIaisle.tolist()
            thirdaisle = VIaisle  # 第三通道的列表
            c = {"thirdaisle": thirdaisle}
            df1 = pd.DataFrame(c, columns=['thirdaisle'])

            # pic 电流通道：电流  电压通道：电压
            pic = pic.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            pic = pic * 255
            # pic1 电流通道：电流  电压通道：IPW
            pic1 = pic1.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            pic1 = pic1 * 255
            # pic2 电流通道：无功电流  电压通道：电压
            pic2 = pic2.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
            pic2 = pic2 * 255

            PIC = pd.concat([pic['current'], pic['voltage'], df1], axis=1)

            # x = range(0, len(pic['current']))
            # plt.plot(x, pic['current'])
            # plt.show()
            # #
            # x = range(0, len(pic['voltage']))
            # plt.plot(x, pic['voltage'])
            # plt.show()
            #
            # x = range(0, len(pic2['current']))
            # plt.plot(x, pic2['current'])
            # plt.show()
            #
            # x = range(0, len(pic2['voltage']))
            # plt.plot(x, pic2['voltage'])
            # plt.show()

            aa = datatomatrix(PIC)
            img = Image.fromarray(np.uint8(aa)).convert('RGB')  # 将数组转化回图片
            img.save(save_path + "{}.png".format(pic_name))  # 将数组保存为图片


    # 图像添加水印
    def lsb_encoder(copyright_image, original_image):
        """
        add copyright image into original image by LSB

        :param copyright_image: RGB image with numpy type
        :param original_image: RGB image with numpy type
        :return: the image that had been processd by LSB and informations
        """
        # 1: 确保输入图像为8bits无符号整型
        original_image = original_image.astype(np.uint8)
        copyright_image = copyright_image.astype(np.uint8)

        # 2: 对original图像和copyright图像备份，不能在原图上更改
        watermark = original_image.copy()
        copyright = copyright_image.copy()

        # 3：将copyright二值化，使其仅含0和1，用1bit表示
        copyright[copyright < 200] = 1
        copyright[copyright >= 200] = 0

        # 4：将watermark的最后1bit的R、G、B全部置零
        #     也可以仅仅对R通道置零
        for i in range(0, watermark.shape[0]):
            for j in range(0, watermark.shape[1]):
                watermark[i, j, :] = (watermark[i, j, :] // 2) * 2

        for i in range(0, copyright.shape[0]):
            for j in range(0, copyright.shape[1]):
                # 5：将用1bit表示的二值化的水印信息
                #   添加到watermark最后1bit上
                watermark[i, j, 0] = watermark[i, j, 0] + copyright[i, j, 0]

        return watermark


    # 加载label和图片
    # pic, label = create_data()
    # pic = pic.reshape(-1, 20, 25, 3)
    # pic = torch.from_numpy(np.transpose(pic, (0, 3, 1, 2)))
    # x_test = torch.tensor(total_test_images, dtype=torch.float32)
    # x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    # x_test = pic.to(device)
    # y_test = torch.tensor(label, dtype=torch.long).to(device)

    # batch_size = 1
    # batch_num_test = int(x_test.shape[0] / batch_size)

    # 实例化网络
    net = PyTorch_Net()
    model_dict = torch.load("vi_model/cnn_model.t")
    net.load_state_dict(model_dict['net'])
    net.eval()
    net = net.to(device)

    prediect('./pic - 空副本/testtest/test1.png')