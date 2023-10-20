import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch import nn

import SCSO
from SCSO import *
from basic_BP import basic_bp as basic_bp
from basic_BP import basic_bp_weight_modified as bp_weight
import GET_DATA
from train_and_valid import train_and_valid
import set_random_seed
import time

if __name__ == '__main__':
    # time1=time.time()
    set_random_seed.get_random_seed(88)
    # train_DataLoader,test_DataLoader,data_train,data_train_target,data_test,data_test_target,feature_num=GET_DATA.GET_DATA(r'data.xlsx',['train','test'],[0,1,2,3,4,5,6],64)
    train_DataLoader,data_train,data_train_target,data_test,data_test_target,feature_num=GET_DATA.GET_510(r'510.xlsx', ['train'], [0, 1, 2, 3, 4, 5, 6], 64)
    # print(feature_num)
    # print(data_train.shape, data_train_target.shape, data_test.shape, data_test_target.shape)


    input_num=feature_num
    hidden_num=14  ##设定隐藏层是4
    output_num=1
    dim=feature_num*hidden_num + hidden_num + hidden_num * output_num + output_num
    # print(dim)











    S=[ i/10 for i in range(10,600)]

    best_test_acc=0
    for s in S:###bp with TSO
        loss_fn_SCSO=nn.BCELoss()

        Best_Score,BestFit,Convergence_curve=SCSO(1000,1000,torch.ones(dim)*(-10),torch.ones(dim)*10,dim,fobj,[input_num,hidden_num,output_num],loss_fn_SCSO,data_train,data_train_target,s)

        print(BestFit)
        print('--------BEST_FIT-----------------')
        print(Best_Score)
        print('--------BEST_SCORE-----------------')
        print(Convergence_curve)


        model_SCSO=bp_weight(input_num, hidden_num, output_num,BestFit)

        # print(list(model_TSO.parameters()))

        opt_SCSO = torch.optim.SGD(model_SCSO.parameters(), lr=0.001)
        loss_SCSO = nn.BCELoss()
        epochs_SCSO=500

        tso_loss_train,tso_loss_test,tso_acc_train,tso_acc_test,probility=train_and_valid(model_SCSO, opt_SCSO, loss_SCSO,epochs_SCSO, train_DataLoader, data_train, data_train_target, data_test,data_test_target)
        if best_test_acc<=tso_acc_test:
            best_test_acc=tso_acc_test
            with open('res.txt', 'a') as f:
                print(tso_acc_train,file=f)
                print(tso_acc_test,file=f)
                print(probility, file=f)



    #     if best_score < tso_acc_test:
    #         best_probility=probility
    #         best_score=tso_acc_test
    #         print(i)
    # print(best_score)
    # print(best_probility)







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


