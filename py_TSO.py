import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
data_from_xlsx=pd.read_excel(r'data.xlsx',sheet_name=(['train','test']),usecols=[0,1,2,3,4,5,6],)
data_train=data_from_xlsx['train']  ##.value numpy
data_test=data_from_xlsx['test']
data_train_target=data_train['target'].values
data_test_target=data_test['target'].values
data_train=data_train[['a','b','c','d','e','f']].values
data_test=data_test[['a','b','c','d','e','f']].values
data_train_target=torch.from_numpy(data_train_target)
data_test_target=torch.from_numpy(data_test_target)
print(data_train_target.shape)

def get_data(train_data,test_data,train_data_target,test_data_target,batch_size):
    train_ds=TensorDataset(train_data,train_data_target)
    test_ds = TensorDataset(test_data, test_data_target)
    return DataLoader(train_ds,batch_size=batch_size,shuffle=True),DataLoader(test_data,batch_size=batch_size)


def loss_batch(model,loss_func,x_batch,y_batch,opt=None):
    loss=loss_func(model(x_batch),y_batch)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(),len(x_batch)


def fit(steps,model,loss_func,opt,train_dl,test_dl):
    for step in range(steps):
        model.train()
        for x_batch,y_batch in train_dl:
            loss_batch(model,loss_func,x_batch,y_batch,opt)

        model.eval()
        with torch.no_grad():
            losses,nums=zip(*[loss_batch(model,loss_func,x_batch,y_batch) for x_b,y_b in test_dl])
        test_loss=np.sum(np.multiply(losses,nums))/np.sum(nums)
        print('当前step: '+str(step),'测试集损失: '+str(test_loss))

class basic_bp(nn.Module):
    def __init__(self):
        super(basic_bp, self).__init__()
        self.fc1=nn.Linear(6,4)
        # self.relu=nn.ReLU()
        self.fc2=nn.Linear(4,1)
        # self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=F.relu(self.fc1(x))

        x=F.sigmoid(self.fc2(x))

        return x


model=basic_bp()
