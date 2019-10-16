#Filename:	main.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 07 Okt 2019 02:30:23  WIB

import argparse
import os
import copy
import torch.optim as optim
from torchvision import transforms
from datetime import datetime
from tensorboardX import SummaryWriter 
from train import *
from utils.cifar100 import *
from model.vgg19 import *

def parse_param():
    """
    parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cls", type = int, default = 100, help = "dataset classes")
    parser.add_argument("-gpu", type = bool, default = True, help = "Use gpu to accelerate")
    parser.add_argument("-batch_size", type = int, default = 64, help = "batch size for dataloader")
    parser.add_argument("-lr", type = float, default = 0.1, help = "initial learning rate")
    parser.add_argument("-epoch", type = int, default = 100, help = "training epoch")
    parser.add_argument("-attention", type = bool, default = True, help = "Attention mechanism")
    parser.add_argument("-optimizer", type = str, default = "sgd", help = "optimizer")
    parser.add_argument("-decay", nargs = '+', type = int, default = [50, 70, 90], help = "epoch for weight decay")
    args = parser.parse_args()

    return args

def print_param(args):
    """
    print the arguments
    """
    print("-" * 15 + "training configuration" + "-" * 15)
    print("class number:{}".format(args.cls))
    print("batch size:{}".format(args.batch_size))
    print("gpu used:{}".format(args.gpu))
    print("learning rate:{}".format(args.lr))
    print("training epoch:{}".format(args.epoch))
    print("attention or not:{}".format(args.attention))
    print("optimizer used:{}".format(args.optimizer))
    print("weights decay:{}-{}-{}".format(args.decay[0], args.decay[1], args.decay[2]))
    print("-" * 53)

def run(model, train_loader, test_loader, optimizer, loss_func,  writer, train_scheduler, epoch):

    best_acc = 0
    best_top5 = 0
    best_epoch = 0
    best_model = model
    iterations = len(train_loader)

    for i in range(epoch):
        #train_scheduler(epoch)
        train_scheduler.step()
        print("Epoch {}".format(i))
        # performance on training set
        model, train_loss, train_acc, time_elapsed = train(model, train_loader, loss_func, optimizer, True)
        print("Training set: Epoch {}, Loss {}, Accuracy {}, Time Elapsed {}".format(i, train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset), time_elapsed))
        writer.add_scalar("Train/loss", train_loss / len(train_loader.dataset), i)
        writer.add_scalar("Train/acc", train_acc / len(train_loader.dataset), i)
   
        # record the layers' gradient
        for name, param in model.named_parameters():
            if "weight" in name and not isinstance(param.grad, type(None)):
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram("{}/{}_grad".format(layer, attr), param.grad.clone().cpu().data.numpy(), i)

        # record the weights distribution
        for name, param in model.named_parameters():
            if "weight" in name:
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram("{}/{}".format(layer, attr), param.clone().cpu().data.numpy(), i)
        # performance on test set
        test_loss, test_acc, top5, time_elapsed = test(model, test_loader, loss_func, True)
        print("Test set: Epoch {}, Loss {}, Accuracy {}, Top 5 {}, Time Elapsed {}".format(i, test_loss / len(test_loader.dataset), test_acc / len(test_loader.dataset), top5 / len(test_loader.dataset), time_elapsed))
        writer.add_scalar("Test/loss", test_loss / len(test_loader.dataset), i)
        writer.add_scalar("Test/acc", test_acc / len(test_loader.dataset), i)
        writer.add_scalar("Test/top5", top5 / len(test_loader.dataset), i)
    
        test_acc = float(test_acc) / len(test_loader.dataset)
        top5 = float(top5) / len(test_loader.dataset)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            best_iters = i

        if top5 > best_top5:
            best_top5 = top5

    return best_model, best_acc, best_top5, best_iters

if __name__ == "__main__":

    args = parse_param()
    print_param(args)
    print(args)

    train_list = "/home/yongjie/code/HieCNN/dataset/CIFAR100/cifar-100-python/train"
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((32, 32), padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                             std=[0.2673, 0.2564, 0.2762]),
        ])
    cifar100_train = CIFAR(train_list, train_transform)
    train_loader = Data.DataLoader(dataset = cifar100_train, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    test_list = "/home/yongjie/code/HieCNN/dataset/CIFAR100/cifar-100-python/test"
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                             std=[0.2673, 0.2564, 0.2762]),
        ])
    # create evaluation data
    cifar100_test = CIFAR(test_list, test_transform)
    test_loader = Data.DataLoader(dataset = cifar100_test, batch_size = args.batch_size, shuffle = True, num_workers = 20)

    # specify the loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # specify the model
    model = VGG(32, 100, attention = True)

    # specify gpu used
    if args.gpu == True:
        model = model.cuda()
        loss_func = loss_func.cuda()
    
    # specify optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 0.0001)
    else:
        optimizer = optim.momentum(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 0.002)
    
    # specify the epoches
    epoch = args.epoch

    # specify the weights decay
    milestones = args.decay

    Time = "{}".format(datetime.now().isoformat(timespec='seconds')).replace(':', '-')
    writer = SummaryWriter(log_dir = os.path.join("./log/", Time))
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = 0.1)
    best_model, best_acc, best_top5, best_iters = run(model, train_loader, test_loader, optimizer, loss_func, writer, train_scheduler, epoch)
    print("Best acc {} at iteration {}, Top 5 {}".format(best_acc, best_iters, best_top5))

    # save model
    model_name = os.path.join("weights", "_" + str(best_iters) + ".pkl")
    torch.save(best_model, model_name)

