import math

import torch
import copy
from basic_BP import basic_bp_weight_modified as bp_weight
device=torch.device('cuda:1')
def initialization(SearchAgents_no, dim, ub, lb):  #ub格式为 []
    #     SearchAgents_no 为种群数量
    # Boundary_no = ub.shape[0]   ##比如ub [1,1,1]
    Positions=torch.rand(SearchAgents_no,dim)*(ub-lb)+lb         ##核实一下
    return Positions ##Positions [几个种群 几个需要优化的参数]



def TAO(Particles_no,Max_iter,UP,LOW,Dim,fojb,struct_of_model,loss_fn,data_train,data_train_target):  ##原理 fobj=  loss (model(data_train),data_train_target)

    Tunal=torch.zeros(Dim)  ##存最优参数
    Tunal_fit=torch.inf   ##存最佳适应度 适应度越低越好
    T = initialization(Particles_no,Dim,UP,LOW);
    Iter=0
    aa=0.75 #aa是一个常数，用于确定金枪鱼跟随的最优个体和上一个个体的比重
    z=0.05 ##超参数
    Convergence_curve_iter0_to_MaxIter = []
    fitness=[ torch.inf for i in range(Particles_no)]
    # best_Tunal=torch.zeros(Dim)
    # best_Tunal_fit=torch.inf
    while Iter<Max_iter:

        C=Iter/Max_iter
        a1=aa+(1-aa)*C
        a2=(1-aa)-(1-aa)*C

        # fitness = []  ##计算某个iter  全部种群Particles_no的最佳适应度
        for i in range(Particles_no):
            Flag4ub=T[i]>UP  ##判断positions是否超过边界
            Flag4lb=T[i]<LOW
            T[i]=T[i]*(~(Flag4lb+Flag4lb)) + UP*Flag4ub + LOW*Flag4lb    ##T 就是positions数组 [Particles_no,DIM]

            # fitness.append(fobj(T[i]))
            fitness[i]=fojb(T[i],struct_of_model,loss_fn,data_train,data_train_target)

            if fitness[i]<Tunal_fit:
                Tunal_fit=fitness[i]

                Tunal=copy.deepcopy(T[i])   ##应该是硬拷贝 不同于matlab



        if Iter==0:
            fit_old=copy.deepcopy(fitness) ##
            C_old=copy.deepcopy(T)

        for i in range(Particles_no):    ##选取最好的fit 适应度 存入当前fitness
            if fit_old[i]<fitness[i]:
                fitness[i]=fit_old[i]
                T[i]=C_old[i]

        C_old=copy.deepcopy(T)
        fit_old=copy.deepcopy(fitness)


        t=(1-Iter/Max_iter)**(Iter/Max_iter)  ###原文中的p

        ##rand---- 随机均匀分布0-1之间的数
        ##这里的代码是进行第      一       个族群时候进行的
        if torch.rand(1).item()<z:
            T[0]=(UP-LOW)*torch.rand(Dim) + LOW    ## matlab 代码里的是乘一个数
        else:     ### >=Z
            if torch.rand(1).item()<0.5:   ##五五开
                r1=torch.rand(1).item()
                ##与matlab 源代码有疑问  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Beta=math.exp(r1*math.exp(3*math.cos(math.pi*((Max_iter-Iter+1)/Max_iter)))) *math.cos(2*math.pi*r1)
                if C>torch.rand(1).item():
                    T[0]=a1*(Tunal+Beta* (  (Tunal-T[0]).abs() ) )  + a2*T[0]
                else:
                    IndivRand=torch.rand(Dim)*(UP-LOW)+LOW
                    T[0]=a1*(IndivRand+ Beta* ((IndivRand-T[0]).abs())  ) + a2*T[0]

            else:
                TF= (torch.rand(1).item()<0.5)*2-1
                if torch.rand(1).item()<0.5:
                    T[0]=Tunal+ torch.rand(Dim)*(Tunal-T[0])+ TF*(t**2)*(Tunal-T[0])


                else:
                    T[0]=TF*(t**2)*T[0]


        for i in range(1,Particles_no):
            if torch.rand(1).item() < z:
                T[i] = (UP - LOW) * torch.rand(Dim) + LOW  ## matlab 代码里的是乘一个数
            else:  ### >=Z
                if torch.rand(1).item() < 0.5:  ##五五开
                    r1 = torch.rand(1).item()
                    ##与matlab 源代码有疑问  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Beta = math.exp(r1 * math.exp(3 * math.cos(math.pi * ((Max_iter - Iter + 1) / Max_iter)))) * math.cos( 2 * math.pi * r1)
                    if C > torch.rand(1).item():
                        T[i] = a1 * (Tunal + Beta * ((Tunal - T[0]).abs())) + a2 * T[i-1]
                    else:
                        IndivRand = torch.rand(Dim) * (UP - LOW) + LOW
                        T[i] = a1 * (IndivRand + Beta * ((IndivRand - T[i]).abs())) + a2 * T[i-1]

                else:
                    TF = (torch.rand(1).item() < 0.5) * 2 - 1
                    if torch.rand(1).item() < 0.5:
                        T[i] = Tunal + torch.rand(Dim) * (Tunal - T[i]) + TF * (t ** 2) * (Tunal - T[i])


                    else:
                        T[i] = TF * (t ** 2) * T[i]
        Convergence_curve_iter0_to_MaxIter.append(Tunal_fit)
        Iter=Iter+1

    # print(Tunal_fit,Tunal,fojb(Tunal,struct_of_model,loss_fn,data_train,data_train_target))
    return Convergence_curve_iter0_to_MaxIter,Tunal,Tunal_fit     ###存 每个iter里面最佳的参数 [dim]




def fojb(vec_tensor,struct_of_model,loss_fn,data_train,data_train_target):
    input_num=struct_of_model[0]
    hidden_num=struct_of_model[1]
    output_num=struct_of_model[2]
    model=bp_weight(input_num,hidden_num,output_num,vec_tensor)  ##创建一个模型 模型的原始参数用 T中的位置

    model=model.to(device)

    # print(model)
    model.eval()


    loss_fn=loss_fn.to(device)
    data_train=data_train.to(device)
    data_train_target=data_train_target.to(device)

    with torch.no_grad():
        loss_train = loss_fn(model(data_train).squeeze(-1), data_train_target).item()


    return loss_train





# A,B,C=TAO(1000,100,torch.tensor([1,1,1]),torch.tensor([-1,-1,-1]),3,fojb)   #(Particles_no,Max_iter,UP,LOW,Dim,fobj):


# print(A,'    ',B,'    ',C,'    ',fojb(B))






###matlab 源代码中 Beta里和论文中的描述有出入 这个版本的TAO遵循 论文中的公式
def TAO_obey_the_easy(Particles_no,Max_iter,UP,LOW,Dim,fojb,struct_of_model,loss_fn,data_train,data_train_target):
    Tunal = torch.zeros(Dim)  ##存最优参数
    Tunal_fit = torch.inf  ##存最佳适应度 适应度越低越好
    T = initialization(Particles_no, Dim, UP, LOW);
    Iter = 0
    aa = 0.75  # aa是一个常数，用于确定金枪鱼跟随的最优个体和上一个个体的比重
    z = 0.05  ##超参数
    Convergence_curve_iter0_to_MaxIter = []
    fitness = [torch.inf for i in range(Particles_no)]
    # best_Tunal=torch.zeros(Dim)
    # best_Tunal_fit=torch.inf
    while Iter < Max_iter:

        C = Iter / Max_iter
        a1 = aa + (1 - aa) * C
        a2 = (1 - aa) - (1 - aa) * C

        # fitness = []  ##计算某个iter  全部种群Particles_no的最佳适应度
        for i in range(Particles_no):
            Flag4ub = T[i] > UP  ##判断positions是否超过边界
            Flag4lb = T[i] < LOW
            T[i] = T[i] * (~(Flag4lb + Flag4lb)) + UP * Flag4ub + LOW * Flag4lb  ##T 就是positions数组 [Particles_no,DIM]

            # fitness.append(fobj(T[i]))
            fitness[i] = fojb(T[i], struct_of_model, loss_fn, data_train, data_train_target)

            if fitness[i] < Tunal_fit:
                Tunal_fit = fitness[i]

                Tunal = copy.deepcopy(T[i])  ##应该是硬拷贝 不同于matlab

        if Iter == 0:
            fit_old = copy.deepcopy(fitness)  ##
            C_old = copy.deepcopy(T)

        for i in range(Particles_no):  ##选取最好的fit 适应度 存入当前fitness
            if fit_old[i] < fitness[i]:
                fitness[i] = fit_old[i]
                T[i] = C_old[i]

        C_old = copy.deepcopy(T)
        fit_old = copy.deepcopy(fitness)

        t = (1 - Iter / Max_iter) ** (Iter / Max_iter)  ###原文中的p

        ##rand---- 随机均匀分布0-1之间的数
        ##这里的代码是进行第      一       个族群时候进行的
        if torch.rand(1).item() < z:
            T[0] = (UP - LOW) * torch.rand(Dim) + LOW  ## matlab 代码里的是乘一个数
        else:  ### >=Z
            if torch.rand(1).item() < 0.5:  ##五五开
                r1 = torch.rand(1).item()
                ##与matlab 源代码有疑问  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Beta = math.exp(r1 * math.exp(3 * math.cos(math.pi * ((Max_iter - Iter + 1) / Max_iter)))) * math.cos(
                    2 * math.pi * r1)
                if C > torch.rand(1).item():
                    T[0] = a1 * (Tunal + Beta * ((Tunal - T[0]).abs())) + a2 * T[0]
                else:
                    IndivRand = torch.rand(Dim) * (UP - LOW) + LOW
                    T[0] = a1 * (IndivRand + Beta * ((IndivRand - T[0]).abs())) + a2 * T[0]

            else:
                TF = (torch.rand(1).item() < 0.5) * 2 - 1
                if torch.rand(1).item() < 0.5:
                    T[0] = Tunal + torch.rand(Dim) * (Tunal - T[0]) + TF * (t ** 2) * (Tunal - T[0])


                else:
                    T[0] = TF * (t ** 2) * T[0]

        for i in range(1, Particles_no):
            if torch.rand(1).item() < z:
                T[i] = (UP - LOW) * torch.rand(Dim) + LOW  ## matlab 代码里的是乘一个数
            else:  ### >=Z
                if torch.rand(1).item() < 0.5:  ##五五开
                    r1 = torch.rand(1).item()
                    ##与matlab 源代码有疑问  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Beta = math.exp(
                        r1 * math.exp(3 * math.cos(math.pi * ((Max_iter - Iter + 1) / Max_iter)))) * math.cos(
                        2 * math.pi * r1)
                    if C > torch.rand(1).item():
                        T[i] = a1 * (Tunal + Beta * ((Tunal - T[0]).abs())) + a2 * T[i - 1]
                    else:
                        IndivRand = torch.rand(Dim) * (UP - LOW) + LOW
                        T[i] = a1 * (IndivRand + Beta * ((IndivRand - T[i]).abs())) + a2 * T[i - 1]

                else:
                    TF = (torch.rand(1).item() < 0.5) * 2 - 1
                    if torch.rand(1).item() < 0.5:
                        T[i] = Tunal + torch.rand(Dim) * (Tunal - T[i]) + TF * (t ** 2) * (Tunal - T[i])


                    else:
                        T[i] = TF * (t ** 2) * T[i]
        Convergence_curve_iter0_to_MaxIter.append(Tunal_fit)
        Iter = Iter + 1

    # print(Tunal_fit,Tunal,fojb(Tunal,struct_of_model,loss_fn,data_train,data_train_target))
    return Convergence_curve_iter0_to_MaxIter, Tunal, Tunal_fit  ###存 每个iter里面最佳的参数 [dim]


















