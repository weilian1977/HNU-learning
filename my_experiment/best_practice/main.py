#coding:utf8
from typing import Tuple
from config import opt
import time
import os
import models
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchnet import meter
# from utils.visualize import Visualizer
from tqdm import tqdm
import torchvision
from threading import Thread

loop = True

def DatasetGet(datapath: str, is_train: bool) -> torch.utils.data.Dataset:
    # 定义对数据的预处理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # 转为Tensor
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                         (0.5, 0.5, 0.5)),  # 归一化
    ])

    dataset = None
    if is_train:
        # 训练集
        dataset = torchvision.datasets.CIFAR10(root=datapath,
                                               train=True,
                                               download=True,
                                               transform=transform)
    else:
        # 测试集
        dataset = torchvision.datasets.CIFAR10(datapath,
                                               train=False,
                                               download=False,
                                               transform=transform)
    return dataset


@torch.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt._parse(kwargs)

    # configure model
    model: models.BasicModule = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)
    model.eval()

    # data
    train_data = DatasetGet(opt.test_data_root, is_train=False)
    test_dataloader = DataLoader(train_data,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        probability = torch.nn.functional.softmax(score,
                                              dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_)
                         for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):
    opt._parse(kwargs)
    # vis = Visualizer(opt.env, port=opt.vis_port)

    # step1: configure model
    model: nn.Module = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)
    model.train()

    # step2: data
    train_data = DatasetGet(opt.train_data_root, is_train=True)
    # val_data = DatasetGet(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data,
                                  opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    # val_dataloader = DataLoader(val_data,
    #                             opt.batch_size,
    #                             shuffle=False,
    #                             num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr: float = opt.lr
    optimizer: torch.optim.Optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4: meters
    # loss_meter = meter.AverageValueMeter()
    # confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    # train
    for epoch in range(opt.max_epoch):

        # loss_meter.reset()
        # confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # train model
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            # loss_meter.add(loss.item())
            # detach 一下更安全保险
            # confusion_matrix.add(score.detach(), target.detach())

            # if (ii + 1) % opt.print_freq == 0:
            #     vis.plot('loss', loss.item())

            #     # 进入debug模式
            #     if os.path.exists(opt.debug_file):
            #         import ipdb
            #         ipdb.set_trace()

        model.save()

        # validate and visualize
        # val_cm, val_accuracy = val(model, val_dataloader)

        # vis.plot('val_accuracy', val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
        #             epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))

        # # update learning rate
        # if loss_meter.value()[0] > previous_loss:
        #     lr = lr * opt.lr_decay
        #     # 第二种降低学习率的方法:不会有moment等信息的丢失
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        # previous_loss = loss_meter.value()[0]


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':

    def noInterrupt():
        while loop:
            import fire
            fire.Fire()

    th = Thread(target=noInterrupt, name="work-thread")
    th.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        loop = False

    print("Wait for thread exit")
    th.join()
    print("End...")
    
