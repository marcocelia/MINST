import pandas as pd
import torch

# define a cross validation function
def crossvalid(*, train, valid, model, criterion, optimizer, dataset, bs, k_fold=5):

    train_score = pd.Series()
    val_score = pd.Series()

    total_size = len(dataset)
    seg = int(total_size/k_fold)
    # tr:train,val:valid; r:right,l:left;  eg: tr_rr: right index of right side train subset
    # index: [tr_ll,tr_lr],[val_l,val_r],[tr_rl,tr_rr]
    for i in range(k_fold):
        tr_ll = 0
        tr_lr = i * seg
        val_l = tr_lr
        val_r = val_l + seg
        tr_rl = val_r
        tr_rr = total_size
        # msg
        # print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)"
        #       % (tr_ll,tr_lr,tr_rl,tr_rr,val_l,val_r))

        train_indices = list(range(tr_ll,tr_lr)) + list(range(tr_rl,tr_rr))
        val_indices = list(range(val_l,val_r))

        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset,val_indices)

        # print(len(train_set),len(val_set))
        # print()

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=4)

        net = model()
        train(net, criterion,optimizer,train_loader,epoch=1)
        val_acc = valid(net, criterion,optimizer,val_loader)
        val_score.at[i] = val_acc

    return val_score
