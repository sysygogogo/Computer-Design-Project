# 训练网络模型 并保存
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import os
import random
from skimage import io
import warnings
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


penalty = 0.0007   #正则化惩罚力度
#speed = 0.001

classifier_data = os.listdir("./pic1_RiGipwBvif")

#classifier_data = ['Arc_Computer', 'Arc_Fan', 'Arc_Hairdryer','Computer', 'Fan', 'Hairdryer']
# classifier_data = ['Arc_Fan', 'Arc_Phone',  'Computer', 'Phone', ]
print(classifier_data)

'''处理图像数据:将图片读为像素值，变成一列数组，分别加入列表中'''
def load_image(path):
    """
    load images from directory to return a list type of data
    其中第一个dirpath(string)是搜索目录，
    第二个dirnames(list)为搜索目录下的所有文件夹，
    第三个filenames(list)为搜索目录下所有的文件。
    """
    img_list = []
    for dirpath, dirname, filenames in os.walk(path):
        for filename in filenames:
            img = io.imread(path + filename) #读取图片  size(20,25,3)
            #print(img)    #打印图片的像素值
            img_flat = np.reshape(img, (1, -1)) #将图片像素值转换成一行  长度(1500)
            #print(img_flat)
            img_list.append(img_flat)#追加单个元素到List的尾部，List中存入一个个的数组  [[],[],[],...]
    return img_list  # [[],[],[],...,[]]


'''生成图像数组'''
def create_data():
    total_images_list = [[] for i in range(len(classifier_data))]
    total_labels_list = [[] for i in range(len(classifier_data))]

    _total_train_images_list = [[] for i in range(len(classifier_data))]
    _total_train_labels_list = [[] for i in range(len(classifier_data))]
    _total_test_images_list = [[] for i in range(len(classifier_data))]
    _total_test_labels_list = [[] for i in range(len(classifier_data))]

    train_images_list = []
    test_images_list = []
    train_labels_list = []
    test_labels_list = []


    for i in range(len(classifier_data)): # 0-10
        data_path = "./pic1_RiGipwBvif/%s/" % classifier_data[i]
        data = load_image(data_path)      # 每次data包含了一个文件夹的数据 [[],[],[],...,[]]
        print(len(data))
        letter = [0 for _ in range(len(classifier_data))]
        letter[i] = 1
        for j in range(len(data)):
            total_images_list[i].extend(np.array(data[j], dtype=np.float32))
            total_labels_list[i].append(letter)

        # 将每一个类别的列表划分数据集
        _total_train_images_list[i], _total_test_images_list[i], _total_train_labels_list[i], _total_test_labels_list[i] = train_test_split(total_images_list[i], total_labels_list[i], test_size=0.3, random_state=4 ,shuffle=True) # ,


    for i in range(len(classifier_data)):
        # 合并所有训练集、数据集的列表
        train_images_list = train_images_list + _total_train_images_list[i]
        test_images_list = test_images_list + _total_test_images_list[i]
        train_labels_list = train_labels_list + _total_train_labels_list[i]
        test_labels_list = test_labels_list + _total_test_labels_list[i]

    total_image = train_images_list + test_images_list
    total_label = train_labels_list + test_labels_list
    print(len(total_image))
    print(len(total_label))

    train_images,test_images, train_labels, test_labels = train_test_split(total_image, total_label, test_size=0.3, random_state=4, shuffle=True)  #

    train_images = np.array(train_images, dtype=np.float32)
    test_images = np.array(test_images, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)

    print(train_images.shape)
    print(test_images.shape)
    # print(train_labels)
    print(train_labels.shape)
    print(test_labels.shape)
    return train_images, test_images, train_labels, test_labels


##################################################################################### 网络模型
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
#####################################################################################


# 神经网络实例化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion_id = nn.CrossEntropyLoss()
criterion_id.to(device)
net = PyTorch_Net()
net.to(device)
cudnn.benchmark = True
log_dir = 'log/' + '/'
# if not os.path.isdir(log_dir):
#     os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

# 反向传播优化器选择: SGD or Adam
optimizer_ExpLR = torch.optim.Adam(net.parameters(), lr=0.001)
# 写了两种学习率更新方式: 指数衰减(原有方式) and 固定步长衰减
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_ExpLR, gamma=0.98)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ExpLR, step_size=5000, gamma=0.5, last_epoch=-1)


total_train_images, total_test_images, total_train_labels, total_test_labels = create_data()
# (751,1500)         (323,1500)          (751,11)            (323,11)

# 准备训练集以及标签
# (751,1500)->(751,20,25,3)
total_train_images = total_train_images.reshape(-1,20,25,3)
# (751,20,25,3)->(751,3,20,25)
total_train_images = torch.from_numpy(np.transpose(total_train_images,(0,3,1,2)))
# x_train = torch.tensor(total_train_images, dtype=torch.float32)
# x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_train = total_train_images.to(device)

y_train = torch.tensor(total_train_labels, dtype=torch.long).to(device)

# 测试集以及标签
total_test_images = total_test_images.reshape(-1,20,25,3)
total_test_images = torch.from_numpy(np.transpose(total_test_images,(0,3,1,2)))
# x_test = torch.tensor(total_test_images, dtype=torch.float32)
# x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
x_test = total_test_images.to(device)

y_test = torch.tensor(total_test_labels, dtype=torch.long).to(device)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


batch_size = 1
batch_num_train = int(x_train.shape[0] / batch_size)  # 根据上面设置的batch_size求得有多个batch
print("train batch num: ",batch_num_train)
# 训练
def train():

    net.train()
    start_num = 0   # 遍历数据集的目录
    train_loss = 0  # 整个epoch的loss
    train_acc = 0   # 整个epoch的acc

    for i in range(0,batch_num_train+1):
        xs = x_train[start_num:start_num+batch_size,:]
        ys = y_train[start_num:start_num+batch_size,:]
        start_num = start_num + batch_size


        x_image = xs.reshape(-1, 3 ,20, 25)

        output = net(x_image)

        # print("output: ",output)
        # print("output: ",output.shape)

        softmax = nn.Softmax(dim=-1)
        prediction = softmax(output)  # 得到预测的标签

        # print(prediction)
        # print("prediction: ",prediction.size())
        # print("ys: ",ys.size())
        # # 一个batch的预测标签的索引与真实索引（类别）对比
        # print("1")
        corrrect_prediction = torch.eq(torch.argmax(prediction, 1), torch.argmax(ys, 1))
        # print(corrrect_prediction.float())
        # print(corrrect_prediction.shape)

        # 一个batch的正确率
        accuracy_batch = torch.mean(corrrect_prediction.float())
        # print("accuracy_batch",accuracy_batch)
        # print(accuracy_batch.shape)

        # 正则化
        l2_reg_loss = torch.tensor(0.)
        for param in net.parameters():
            l2_reg_loss = l2_reg_loss.cuda()
            l2_reg_loss += penalty * torch.norm(param.cuda())

        # 计算损失
        # print("ys: ", ys)
        # print("ys: ",ys.size())
        # print("argmax ys:",torch.argmax(ys, 1))

        loss_id = criterion_id(output, torch.argmax(ys, 1)) # bp损失
        loss_batch = loss_id + l2_reg_loss
        optimizer_ExpLR.zero_grad()
        loss_batch.backward()
        optimizer_ExpLR.step()

        train_loss += loss_batch.item() * ys.size(0)
        train_acc = train_acc + accuracy_batch * ys.size(0)
        # print("loss:",train_loss)
        # print("acc:",train_acc)
    # print(train_loss)
    # print(train_acc)
    train_loss = train_loss / x_train.shape[0]
    train_acc = train_acc / x_train.shape[0]

    return train_loss,train_acc


# 测试
batch_num_test = int(x_test.shape[0] / batch_size)

def tet():
    net.eval()
    start_num = 0  # 遍历数据集的目录
    test_loss = 0  # 整个epoch的loss
    test_acc = 0   # 整个epoch的acc
    for i in range(0, batch_num_test + 1):
        xs = x_test[start_num:start_num + batch_size, :]
        ys = y_test[start_num:start_num + batch_size, :]
        start_num = start_num + batch_size

        x_image = xs.reshape(-1, 3 ,20, 25)
        output = net(x_image)

        softmax = nn.Softmax(dim=-1)
        prediction = softmax(output)  # 得到预测的标签

        # 一个batch的预测标签的索引与真实索引（类别）对比
        corrrect_prediction = torch.eq(torch.argmax(prediction, 1), torch.argmax(ys, 1))
        # 一个batch的正确率
        accuracy_batch = torch.mean(corrrect_prediction.float())

        # 正则化
        l2_reg_loss = torch.tensor(0.)
        for param in net.parameters():
            l2_reg_loss = l2_reg_loss.cuda()
            l2_reg_loss += penalty * torch.norm(param)

        # 计算损失
        loss_id = criterion_id(output, torch.argmax(ys, 1))  # bp损失
        loss_batch = loss_id + l2_reg_loss

        # optimizer_ExpLR.zero_grad()
        # loss_batch.backward()
        # optimizer_ExpLR.step()

        test_loss += loss_batch.item()* ys.size(0)
        test_acc = test_acc + accuracy_batch * ys.size(0)
        # print("test loss:",test_loss)
        # print("test acc:",test_acc)

    test_loss = test_loss / x_test.shape[0]
    test_acc = test_acc / x_test.shape[0]

    return test_loss,test_acc


if __name__ == '__main__':
     train_loss_list = []
     train_acc_list = []
     test_loss_list = []
     test_acc_list = []
     for epoch in range(0, 30001):
         # print(epoch)
         train_loss, train_acc = train()
         test_loss, test_acc = tet()

         train_loss_list.append(train_loss)
         train_acc_list.append(train_acc.item())
         test_loss_list.append(test_loss)
         test_acc_list.append(test_acc.item())

         # pred0.extend(pred)
         # labels0.extend(labels)

         # print("训练次数=%f,训练集损失值=%f,训练集准确率=%f,测试集损失值=%f,测试集准确率=%f" % (epoch, train_loss, train_acc, test_loss, test_acc))

         if epoch %1000 ==0:
             print("训练次数=%f,训练集损失值=%f,训练集准确率=%f,测试集损失值=%f,测试集准确率=%f" % (epoch, train_loss, train_acc, test_loss, test_acc))

         if test_acc > 0.97 and test_acc < 0.987:
             print("step", epoch, " now train acc is :", train_acc.item(), " now test acc is :", test_acc.item())
         # if train_acc>=0.99  and test_acc >=0.986 :
         #     print('Saving..')
         #     state = {
         #         'net': CNN_net.state_dict(),
         #     }
         #     if not os.path.isdir('checkpoint'):
         #         os.mkdir('checkpoint')
         #     torch.save(state, './checkpoint/model.pth')
         #     break;

         if train_acc > 0.99 and test_acc > 0.975:
             checkpoint_path = './vi_model/'
             state = {
                 'net': net.state_dict(),
                 'train_acc': train_acc,
                 'test_acc': test_acc,
             }
             torch.save(state, checkpoint_path + 'cnn_model.t')
             print("step", epoch, "now test acc is :", test_acc)
             print("模型保存成功")
             # break

         scheduler.step()
     print("训练集损失", train_loss_list)
     print("训练集准确率", train_acc_list)
     print("测试集损失", test_loss_list)
     print("测试集准确率", test_acc_list)
     # print(type(train_loss_list))
     # print(type(train_acc_list))
     # print(type(test_loss_list))
     # print(type(test_acc_list))
     df_train_loss_list = pd.DataFrame(train_loss_list, columns=['train_loss'])
     df_train_acc_list = pd.DataFrame(train_acc_list, columns=['train_acc'])
     df_test_loss_list = pd.DataFrame(test_loss_list, columns=['test_loss'])
     df_test_acc_list = pd.DataFrame(test_acc_list, columns=['test_acc'])

     df_hebing = pd.concat([df_train_loss_list, df_train_acc_list, df_test_loss_list, df_test_acc_list], axis=1)
     print(df_hebing)
     df_hebing.to_csv('./test400.csv')
     # classifier = train()



# if __name__ == "__main__":  # 模型测试部分
#    model = PyTorch_Net()
#    # model.eval()
#    oritensor = torch.randn(32, 3, 20, 25)
#    newtensor = model(oritensor)







xs = x_train
ys = y_train

x_image = xs.reshape(-1, 3, 25, 20)

output = net(x_image)  # 长11的标签

softmax = nn.Softmax(dim=-1)
prediction = softmax(output)
corrrect_prediction = torch.equal(torch.argmax(prediction, 1), torch.argmax(ys, 1)) # 一个batch的预测标签的索引与真实索引（类别）对比
# 一个batch的正确率
accuracy = torch.mean(torch.tensor(corrrect_prediction.__float__()))

# 正则化
l2_reg_loss = torch.tensor(0.)
for param in net.parameters():
    l2_reg_loss = l2_reg_loss.cuda()
    l2_reg_loss += penalty * torch.norm(param)

# 计算损失
loss_id = criterion_id(output, ys.float())  # bp损失
loss = loss_id + l2_reg_loss
optimizer_ExpLR.zero_grad()
loss.backward()
optimizer_ExpLR.step()

# 收集变量
writer.add_scalar('total_loss', loss)
writer.add_scalar('id_loss', loss_id)
writer.add_scalar('l2_reg_loss', l2_reg_loss)
writer.add_scalar('acc', accuracy)



# # 以下代码只修改了模型保存部分
# round_num = 128
# max_test_acc = 0
# max_testtrain_acc = 0
# print("开始训练")
# for j in range(30001):
#     batch_xs = []
#     batch_ys = []
#
#     for i in range(round_num):
#         rand_num = random.randint(0, total_train_images.shape[0] - 1)
#         batch_xs.append(total_train_images[rand_num])
#         batch_ys.append(total_train_labels[rand_num])
#
#     h_fc1_drop_, cost, _ = sess.run([h_fc1_drop_, cross_entropy, train_step],
#                                     feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1.0})
#
#     # 写入每步训练的值
#     summary1 = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1.0})
#     filewriter1.add_summary(summary1, j)
#
#     summary2 = sess.run(merged, feed_dict={xs: total_test_images, ys: total_test_labels, keep_prob: 1.0})
#     filewriter2.add_summary(summary2, j)
#
#     h_fc1_drop_, test_acc, result = sess.run([h_fc1_drop_, accuracy, prediction],
#                                              feed_dict={xs: total_test_images, ys: total_test_labels,
#                                                         keep_prob: 1.0})
#     max_test_acc = max(max_test_acc, test_acc)
#     train_acc = sess.run(accuracy, feed_dict={xs: total_train_images, ys: total_train_labels, keep_prob: 1.0})
#     if max_test_acc != test_acc:
#         max_testtrain_acc = train_acc
#         now_step = j
#         now_cost= cost
#
#     #######################################################################
#     if train_acc > 0.99 and test_acc > 0.987:
#         checkpoint_path = './vi_model/'
#         state = {
#             'net': net.state_dict(),
#             'train_acc': train_acc,
#             'test_acc': test_acc,
#         }
#         torch.save(state, checkpoint_path + 'cnn_model.t')
#         print("step", j, "now test acc is :", test_acc)
#         print("模型保存成功")
#     #######################################################################
#
#     if test_acc > 0.98 and test_acc < 0.987:
#         print("step", j,  "now test acc is :", test_acc)
#
#     if j % 1000 == 0:
#         print("step", j, "train acc is:", train_acc, "test acc is :", test_acc)
#         print("cost is :", cost)
#         print("global_step,learing_rate:",sess.run([global_step,learing_rate]))
#
#
# print("max_test_acc", max_test_acc, "now_train", max_testtrain_acc,"now_step",now_step,"now_cost",now_cost)
#
# return result