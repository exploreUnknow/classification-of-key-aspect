import time
from myDataset import MyDataset
from torch.utils.data import DataLoader
from PMA import Text1biLstm #Text1biLstm #Text1biLstm #Text1biLstm #  
import torch
import torch.nn as nn

import torch.optim as optim
from sklearn.metrics import f1_score


mod = "_1e-4"#""#"_0"#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
##1和5 卷积核大小不一样

#加载数据
# path = "/home/user/Programs/PMA_lihang/data/data_v_type_train.csv"# 老数据,这个数据有问题，重新使用118的数据进行训练
# path = "/home/user/Programs/PMA_lihang/data/data118_v_type_train.csv"

path = f"/home/user/Programs/PMA_lihang/data/myData118_v_type_train{mod}_v2.csv" #新数据

dataset = MyDataset(path)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
#训练模型
# 检查cuda是否可用
model = Text1biLstm()
#损失函数
#将模型放到GPU上
model.to(device)
criterion = nn.CrossEntropyLoss()
#优化器
optimizer = optim.Adam(model.parameters(),lr=0.00001)
#训练
print("start training")
loss_all = []
#初始化准确率
acc = 0

f1_eval = 0

flag = 0#连续两次f1值都没有提升，就停止训练

for epoch in range(2100):
    start_time = time.time()
    loss_ = 0
    model.train()
    for i,data in enumerate(dataloader):
        inputs,labels = data
        #对labels进行one-hot编码，总共6类，现在的值是下标
        #labels = labels[0]
        # print("inshape",inputs.shape)
        #将labels减少一个维度
        labels = torch.squeeze(labels)
        optimizer.zero_grad()
        #将数据放到GPU上
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        # print("outsahpe",outputs.shape)
        # print("labelshape",labels.shape)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        loss_ += loss.item()


    end_time = time.time()
    loss_all.append(loss_)
    print("epoch:%d,step:%d,loss:%f"%(epoch,i,loss_))
    print("time:%f"%(end_time-start_time))


    #验证模型，计算准确率

    if epoch % 50 == 0:
        model.eval()
        # path = "/home/user/Programs/PMA_lihang/data/data_v_type_eval.csv"
        # path = "/home/user/Programs/PMA_lihang/data/data118_att_type_eval.csv"
        path = f"/home/user/Programs/PMA_lihang/data/myData118_v_type_eval{mod}_v2.csv"
        dataset = MyDataset(path)
        dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
        correct = 0
        total = 0

        ##为了计算f1
        y_true = []
        y_pred = []

        print("start eval")


        for i,data in enumerate(dataloader):
            inputs,labels = data

            for label in labels:
                y_true.append(label[0])

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.squeeze(labels)
            outputs = model(inputs)
            predict = torch.max(outputs,1)[1]

            for i in range(len(predict)):
                y_pred.append(predict.cpu()[i])

            total += labels.size(0)
            correct += (predict == labels).sum()
        print("correct:%d,total:%d,acc:%f"%(correct,total,correct/total))
        #如果现在的准确率大于之前的准确率，保存模型

        f1_eval_now = f1_score(y_true,y_pred,average = 'weighted')
        # f1_eval_now = f1_score(y_true,y_pred,average = 'macro')
        print("f1",f1_eval_now)
        # if correct/total > acc:
        if f1_eval_now > f1_eval + 0.005:
            flag = 0
            f1_eval = f1_eval_now
            acc = correct/total
            torch.save(model.state_dict(),f"/home/user/Programs/PMA_lihang/code_lihang/model/myModel118_1_1k{mod}_1bilstm.pth")
            print("*************epoch:",epoch,"****save model***********")
        else:
            print("*************epoch:",epoch,"***************************not save model************************")
            flag += 1
            if flag == 3:
                print("*************epoch:",epoch,"****break***********")
                break





model = Text1biLstm()
# model.load_state_dict(torch.load("/home/user/Programs/PMA_lihang/code_lihang/model/model_1_1k.pth"))
model.load_state_dict(torch.load(f"/home/user/Programs/PMA_lihang/code_lihang/model/myModel118_1_1k{mod}_1bilstm.pth"))
#将模型放到GPU上
model.to(device)

model.eval()
#加载测试数据
path = "/home/user/Programs/PMA_lihang/data/data118_v_type_test.csv"

# path = f"/home/user/Programs/PMA_lihang/data/myData118_v_type_test_1e-4_v2.csv"


dataset = MyDataset(path)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)
#预测，
correct = 0
total = 0
print("start predict")
resoults = []
#总共13类。计算每一类的精确率和召回率
#每一类的预测正确的个数
class_num = 14
corrects = [0 for i in range(class_num)]
#每一类的预测错误的个数
errors = [0 for i in range(class_num)]
#每一类的实际正确的个数
labels = [0 for i in range(class_num)]


##为了计算f1
y_true = []
y_pred = []


for i,data in enumerate(dataloader):
    inputs,labels_ = data

    for label in labels_:
        y_true.append(label[0])


    labels_ = torch.squeeze(labels_)
    #将数据放到GPU上
    inputs = inputs.to(device)
    labels_ = labels_.to(device)

    outputs = model(inputs)
    predict = torch.max(outputs,1)[1]
    resoults.append(predict)
    total += labels_.size(0)
    correct += (predict == labels_).sum()
    #计算每一类的预测正确的个数
    for i in range(len(predict)):
        y_pred.append(predict.cpu()[i])
        if predict[i] == labels_[i]:
            corrects[predict[i]] += 1
        else:
            errors[predict[i]] += 1
    #计算每一类的个数
    for i in range(len(labels_)):
        labels[labels_[i]] += 1
    # break
print("correct:%d,total:%d,acc:%f"%(correct,total,correct/total))
# print("resoults:",resoults)
print("corrects:",corrects)
print("errors:",errors)
print("labels:",labels)
#计算每一类的精确率和召回率
for i in range(len(corrects)):
    if corrects[i] == 0:
        precision = 0
    else:
        precision = corrects[i]/(corrects[i]+errors[i])
    if labels[i] == 0:
        recall = 0
    else:
        recall = corrects[i]/labels[i]
    print("第%d类的精确率为%f,召回率为%f"%(i,precision,recall))
#计算f1值，只计算精确率和召回率都不为0的类
f1 = 0
count = 0
for i in range(len(corrects)):
    if corrects[i] == 0:
        precision = 0
    else:
        count += 1
        precision = corrects[i]/(corrects[i]+errors[i])
    if labels[i] == 0:
        recall = 0
    else:
        recall = corrects[i]/labels[i]
    if precision == 0 or recall == 0:
        continue
    f1 += 2*precision*recall/(precision+recall)
f1 = f1/count
print("f1:",f1)

from sklearn.metrics import f1_score
# print(y_true[0:10])
# print(y_pred[0:10])

f1_ = f1_score(y_true, y_pred, average='weighted')
# f1_ = f1_score(y_true,y_pred,average = 'macro')

print("最新f1",f1_)

