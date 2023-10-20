import math

import torch
import copy
from basic_BP import basic_bp_weight_modified as bp_weight

def initialization(SearchAgents_no, dim, ub, lb):  #ub格式为 []
    #     SearchAgents_no 为种群数量
    Boundary_no = ub.shape[0]
    if Boundary_no == 1:
        Positions=torch.rand(SearchAgents_no,dim)*(ub-lb)+lb         ##核实一下

    if Boundary_no > 1:
        Positions=torch.zeros(SearchAgents_no,dim)
        # print(Positions.shape)
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:,i]=(torch.rand(SearchAgents_no,1)*(ub_i-lb_i)+lb_i).reshape(SearchAgents_no)

    return Positions ##Positions [几个种群 几个需要优化的参数]


def RouletteWheelSelection(p):   ##轮盘赌
    r=torch.rand(1).item()
    s=p.sum()
    # print(s)
    p=p/s
    C=p.cumsum(dim=0)
    teta=0
    for i,v in enumerate(C):
        if r <= v:
            teta=i
            break
    return teta+1

def fobj(vec_tensor,struct_of_model,loss_fn,data_train,data_train_target):
    input_num = struct_of_model[0]
    hidden_num = struct_of_model[1]
    output_num = struct_of_model[2]
    model = bp_weight(input_num, hidden_num, output_num, vec_tensor)  ##创建一个模型 模型的原始参数用 T中的位置
    # print(model)
    model.eval()
    with torch.no_grad():
        res_train = model(data_train).squeeze(-1)

        res_train = (res_train >= 0.5)
        res_train = (res_train == data_train_target)
        acc_train = (res_train.sum() / len(data_train)).item()
    # print(acc_train)
    return acc_train*(-1)


def fobj_test(Positions_i):
    print((Positions_i[0]-3)*(Positions_i[0]-3) + (Positions_i[1]-2)*(Positions_i[1]-2))
    return (Positions_i[0]-4)*(Positions_i[0]-4) + (Positions_i[1]+2)*(Positions_i[1]+2).item()



# Positions=initialization(20,3,torch.tensor([1,10,0]),torch.tensor([3,20,0.5]))
# print(Positions)
def SCSO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,struct_of_model,loss_fn,data_train,data_train_target,S):
    BestFit=torch.zeros(dim)  ##存最优参数
    Best_Score= torch.inf  ##存最佳适应度 适应度越低越好
    Positions = initialization(SearchAgents_no,dim,ub,lb)
    # print(Positions)
    Convergence_curve= []   ##每次迭代的最佳适应度
    t=1        ##当前迭代次数
    p=torch.linspace(1,360,steps=360)
    while t<=Max_iter:
        for i in range(SearchAgents_no):
            Flag4ub=Positions[i]>ub  ##判断positions是否超过边界
            Flag4lb=Positions[i]<lb
            Positions[i]=Positions[i]*(~(Flag4lb+Flag4lb)) + ub*Flag4ub + lb*Flag4lb    ##T 就是positions数组 [Particles_no,DIM]

            # fitness.append(fobj(T[i]))
            fitness=fobj(Positions[i],struct_of_model,loss_fn,data_train,data_train_target)   #fobj_test(Positions[i])    #
            if fitness<Best_Score:
                Best_Score=fitness
                BestFit=copy.deepcopy(Positions[i])

        S=S      ##S is maximum Sensitivity range
        rg=S- ((S*t)/Max_iter)  ##指导R的参数
        for i in range(SearchAgents_no):
            r=torch.rand(1).item()*rg       ##r 每条猫的灵敏度
            R=2*rg*torch.rand(1).item()-rg   ##R系数 控制进行什么阶段操作
            for j in range(dim):
                teta=RouletteWheelSelection(p)
                if  (-1<=R) and (R<=1) :
                    Rand_position= (torch.rand(1)*BestFit[j]-Positions[i,j]).abs()
                    Positions[i,j]=BestFit[j]-r*Rand_position*torch.cos(torch.tensor(teta)/180*torch.pi)
                else:
                    cp=torch.floor(SearchAgents_no*torch.rand(1)+1).int().item()
                    if cp == SearchAgents_no:
                        cp=cp-1
                    CondidatePosition=Positions[cp,:]
                    Positions[i,j]=r*(CondidatePosition[j]-torch.rand(1)*Positions[i,j])
        t=t+1

        Convergence_curve.append(Best_Score)


    return Best_Score,BestFit,Convergence_curve





# a=1
# b=2
# c=3
# d=4
# e=5
#
# Best_Score,BestFit,Convergence_curve=SCSO(300,100,torch.tensor([0,0]),torch.tensor([5,5]),2,a,b,c,d,e)
# print(Best_Score)
# print(BestFit)
# print(Convergence_curve)





