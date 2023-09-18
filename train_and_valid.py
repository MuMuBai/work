import torch

device=torch.device('cuda:1')
def train_and_valid(model, opt, loss_fn, epochs, train_DataLoader, data_train, data_train_target, data_test,data_test_target):
    for epoch in range(epochs):
        model.train()

        model=model.to(device)
        data_train=data_train.to(device)
        data_train_target=data_train_target.to(device)
        data_test=data_test.to(device)
        data_test_target=data_test_target.to(device)
        loss_fn=loss_fn.to(device)


        for x, y in train_DataLoader:

            x=x.to(device)
            y=y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred.squeeze(-1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        loss_train = loss_fn(model(data_train).squeeze(-1), data_train_target).item()
        print('训练集loss:', loss_train)
        res_train = model(data_train).squeeze(-1)
        res_train = (res_train >= 0.5)
        res_train = (res_train == data_train_target)
        acc_train = (res_train.sum() / len(data_train)).item()
        print('训练集acc:', acc_train)

        loss_test = loss_fn(model(data_test).squeeze(-1), data_test_target).item()
        print('测试集loss', loss_test)
        res_test = model(data_test).squeeze(-1)
        res_test = (res_test >= 0.5)
        res_test = (res_test == data_test_target)
        acc_test = (res_test.sum() / len(data_test)).item()
        print('测试集acc:', acc_test)

    return loss_train, loss_test, acc_train, acc_test