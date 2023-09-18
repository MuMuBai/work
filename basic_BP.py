import torch
from torch import nn
import torch.nn.functional as F
class basic_bp(nn.Module):
    def __init__(self,input_num,hidden_num,output_num):
        super(basic_bp, self).__init__()
        self.fc1=nn.Linear(input_num,hidden_num)

        # self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_num,output_num)
        # self.bn = BN = nn.BatchNorm1d(input_num)
        # self.sigmoid=nn.Sigmoid()
        # print(self.fc1.weight)
        # print(self.fc1.bias)
        # print(self.fc2.weight)
        # print(self.fc2.bias)

    def forward(self,x):
        # x=self.bn(x)
        # print(x)
        x=F.relu(self.fc1(x))

        x=F.sigmoid(self.fc2(x))

        return x



 # def train_and_valid(model,opt,loss_fn,epochs,train_DataLoader,data_train,data_train_target,data_test,data_test_target):
 #        for epoch in range(epochs):
 #            model.train()
 #            for x, y in train_DataLoader:
 #                y_pred = model(x)
 #                loss = loss_fn(y_pred.squeeze(-1), y)
 #                opt.zero_grad()
 #                loss.backward()
 #                opt.step()
 #        model.eval()
 #        with torch.no_grad():
 #            loss_train = loss_fn(model(data_train).squeeze(-1), data_train_target).item()
 #            print('训练集loss:', loss_train)
 #            res_train = model(data_train).squeeze(-1)
 #            res_train = (res_train >= 0.5)
 #            res_train = (res_train == data_train_target)
 #            acc_train = (res_train.sum() / len(data_train)).item()
 #            print('训练集acc:', acc_train)
 #
 #            loss_test = loss_fn(model(data_test).squeeze(-1), data_test_target).item()
 #            print('测试集loss', loss_test)
 #            res_test = model(data_test).squeeze(-1)
 #            res_test = (res_test >= 0.5)
 #            res_test = (res_test == data_test_target)
 #            acc_test = (res_test.sum() / len(data_test)).item()
 #            print('测试集acc:', acc_test)
 #
 #        return loss_train,loss_test,acc_train,acc_test


class basic_bp_weight_modified(nn.Module):
    def __init__(self, input_num, hidden_num, output_num,weight):   #  weight is a tensor
        super(basic_bp_weight_modified, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_num)


        # self.relu=nn.ReLU()
        self.fc2 = nn.Linear(hidden_num, output_num)
        # self.bn=BN = nn.BatchNorm1d(input_num)

        # print(self.fc1.weight.shape)
        # print(self.fc1.bias.shape)
        # print(self.fc2.weight.shape)
        # print(self.fc2.bias.shape)
        # self.sigmoid=nn.Sigmoid()
        self.fc1.weight=torch.nn.Parameter(weight[:input_num*hidden_num].reshape(hidden_num,input_num))
        self.fc1.bias=torch.nn.Parameter(weight[input_num*hidden_num:input_num*hidden_num+hidden_num].reshape(hidden_num))
        self.fc2.weight = torch.nn.Parameter(weight[input_num*hidden_num+hidden_num : input_num*hidden_num+hidden_num+hidden_num].reshape(output_num,hidden_num))
        self.fc2.bias = torch.nn.Parameter(weight[input_num*hidden_num+hidden_num+hidden_num:].reshape(output_num))
        # print(self.fc1.weight.shape)
        # print(self.fc1.bias.shape)
        # print(self.fc2.weight.shape)
        # print(self.fc2.bias.shape)
    def forward(self, x):
        x = F.relu(self.fc1(x))

        x = F.sigmoid(self.fc2(x))

        return x