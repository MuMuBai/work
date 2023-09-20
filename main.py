import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn
from TSO import *
from basic_BP import basic_bp as basic_bp
from basic_BP import basic_bp_weight_modified as bp_weight
import GET_DATA
from train_and_valid import train_and_valid
import set_random_seed
import time
device=torch.device('cuda:1')
# print(device)   ## parameters  hidden_num aa z epochs
# print(torch.cuda.device_count())
if __name__ == '__main__':
    time1=time.time()
    set_random_seed.get_random_seed(88)
    # train_DataLoader,test_DataLoader,data_train,data_train_target,data_test,data_test_target,feature_num=GET_DATA.GET_DATA(r'data.xlsx',['train','test'],[0,1,2,3,4,5,6],64)
    train_DataLoader,data_train,data_train_target,data_test,data_test_target,feature_num=GET_DATA.GET_510(r'510.xlsx', ['train'], [0, 1, 2, 3, 4, 5, 6], 64)
    # print(feature_num)
    # print(data_train.shape, data_train_target.shape, data_test.shape, data_test_target.shape)


    input_num=feature_num
    hidden_num=14  ##设定隐藏层是4
    output_num=1
    dim=feature_num*hidden_num + hidden_num + hidden_num * output_num + output_num
    print(dim)





    best_acc_test=0
    best_aa=0
    best_z=0
    for aa in np.arange(0.24, 1, 0.01):

        aa=float(aa)
        for z in np.arange(0.01,1, 0.01):
            z=float(z)





            ###bp with TSO
            loss_fn_TSO=nn.BCELoss()

            Convergence_curve_iter0_to_MaxIter,Tunal,Tunal_fit=TAO(1000,100,torch.ones(dim)*1,torch.ones(dim)*(-1),dim,fojb,[input_num,hidden_num,output_num],loss_fn_TSO,data_train,data_train_target,aa,z)
            '''
            print(Tunal_fit)
            print('--------TSO_tunal-----------------')
            print(Tunal)
            print('--------TSO_tunal-----------------')
            print(Convergence_curve_iter0_to_MaxIter)
            '''

            model_TSO=bp_weight(input_num, hidden_num, output_num,Tunal)

            # print(list(model_TSO.parameters()))

            opt_TSO = torch.optim.SGD(model_TSO.parameters(), lr=0.001)
            loss_TSO = nn.BCELoss()
            epochs_TSO=500
            tso_loss_train,tso_loss_test,tso_acc_train,tso_acc_test=train_and_valid(model_TSO, opt_TSO, loss_TSO,epochs_TSO, train_DataLoader, data_train, data_train_target, data_test,data_test_target)
            print('\n')
            print('aa:',aa,'z',z)
            print('tso_loss_train： ',tso_loss_train,'tso_loss_test： ',tso_loss_test,'tso_acc_train： ',tso_acc_train,'tso_acc_test： ',tso_acc_test)
            print('\n')
            if best_acc_test < tso_acc_test :
                best_acc_test=tso_acc_test
                best_aa=aa
                best_z=z

    print('best----------------------','aa:',best_aa,' z',best_z)


## 18 94     13 96    11 94.7      14 94.7

'''
    model=basic_bp(input_num,hidden_num,output_num)
        # opt=torch.optim.Adam(model.parameters(),lr=0.01)
    opt=torch.optim.SGD(model.parameters(),lr=0.001)
    loss_fn=nn.BCELoss()
    epochs=500
    loss_train,loss_test,acc_train,acc_test=train_and_valid(model,opt,loss_fn,epochs,train_DataLoader,data_train,data_train_target,data_test,data_test_target)
    print(list(model.parameters()))
    print('hidden_num:',hidden_num,' loss_train： ',loss_train,'loss_test： ',loss_test,'acc_train： ',acc_train,'acc_test： ',acc_test)

'''


