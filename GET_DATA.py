import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def GET_DATA(file_name,sheets,usecols,batch_size):
    data_from_xlsx = pd.read_excel(file_name, sheet_name=(sheets), usecols=usecols)
    data_train = data_from_xlsx['train']  ##.value numpy
    data_test = data_from_xlsx['test']
    data_train_target = data_train['target'].values
    data_test_target = data_test['target'].values
    data_train = data_train[['a', 'b', 'c', 'd', 'e', 'f']].values
    data_test = data_test[['a', 'b', 'c', 'd', 'e', 'f']].values
    data_train = torch.from_numpy(data_train).float()
    data_test = torch.from_numpy(data_test).float()
    # print(data_train.size(),data_train.dtype)
    data_train_target = torch.from_numpy(data_train_target).float()
    data_test_target = torch.from_numpy(data_test_target).float()
    # print(data_train_target)
    fearture_num=data_train.shape[-1]

    train_ds = TensorDataset(data_train, data_train_target)
    test_ds = TensorDataset(data_test, data_test_target)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False), DataLoader(test_ds),data_train,data_train_target,data_test,data_test_target,fearture_num



def GET_510(file_name,sheets,usecols,batch_size):
    data_from_xlsx = pd.read_excel(file_name, sheet_name=(sheets), usecols=usecols)
    data_train = data_from_xlsx['train']  ##.value numpy
    # data_test = data_from_xlsx['test']
    data_train_target = data_train['target'].values
    # data_test_target = data_test['target'].values
    data_train = data_train[['a', 'b', 'c', 'd', 'e', 'f']].values
    # data_test = data_test[['a', 'b', 'c', 'd', 'e', 'f']].values
    data= torch.from_numpy(data_train).float()
    data_target = torch.from_numpy(data_train_target).float()
    # print(data.shape,data_target.shape)

    length=data.shape[0]
    shuffle_index = torch.randperm(length)
    # print(shuffle_index)

    data_shuffled=data[shuffle_index]
    data_shuffled_target=data_target[shuffle_index]
    # print(data_shuffled.shape, data_target_shuffled.shape)
    # data_test_target = torch.from_numpy(data_test_target).float()
    # print(data_train_target)
    data_train=data_shuffled[:length*7//10]
    data_train_target=data_shuffled_target[:length*7//10]


    data_test=data_shuffled[length*7//10:]
    data_test_target=data_shuffled_target[length*7//10:]
    fearture_num=data_train.shape[1]

    train_ds = TensorDataset(data_train, data_train_target)
    # test_ds = TensorDataset(data_test, data_test_target)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False),data_train,data_train_target,data_test,data_test_target,fearture_num