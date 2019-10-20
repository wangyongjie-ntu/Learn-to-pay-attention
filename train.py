#Filename:	train.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Rab 31 Jul 2019 09:36:27  +08

from torch.autograd import Variable
import torch
import time


def train(model, train_loader, loss_func, optimizer, is_gpu=True):
    
    model.train()
    start = time.time()
    epoch_loss = 0
    epoch_acc = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        if is_gpu:
            inputs, labels = Variable(batch_x.float().cuda()),Variable(batch_y.long().cuda())
        else:
            inputs, labels = Variable(batch_x.float()),Variable(batch_y.long())

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs[0].data, 1)
        loss = loss_func(outputs[0], labels)
        loss.backward()
        optimizer.step()
        print("train iteration {}, loss {}, acc {}, lr {}".format(step, loss.item(), torch.sum(preds == labels.data).item()/len(batch_x), optimizer.param_groups[0]['lr']))

        epoch_loss += loss.detach().item()
        epoch_acc += torch.sum(preds == labels.data).item()

    end = time.time()
    time_elapsed = end - start

    return model, epoch_loss, epoch_acc, time_elapsed


def test(model, test_loader, loss_func, is_gpu = True):
    start = time.time()
    epoch_loss = 0
    epoch_acc = 0
    top5 = 0
    mask = max((1, 5))
    model.eval()
    for step, (batch_x, batch_y) in enumerate(test_loader):
        # wrap them in Variable
        if is_gpu:
            inputs, labels = Variable(batch_x.float().cuda()),Variable(batch_y.long().cuda())
        else:
            inputs, labels = Variable(batch_x.float()),Variable(batch_y.long())
        outputs = model(inputs)
        _, preds = torch.max(outputs[0].data, 1)
        loss = loss_func(outputs[0], labels)

        epoch_loss += loss.detach().item()
        epoch_acc += torch.sum(preds == labels.data).item()
        _, top5_preds = outputs[0].topk(mask, 1, True, True)
        # compute the top-5 acc
        for i in range(len(batch_x)):
            if labels[i] in top5_preds[i]:
                top5 += 1
        
    end = time.time()
    time_elapsed = end - start

    return epoch_loss, epoch_acc, top5, time_elapsed

