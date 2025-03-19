import time
from PLTDataset import LicensePlateDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MyResBet18 import ModifiedResNet18
from Triplet import TripletAttention
import torch.nn as nn
from torch.optim import Adam
import train_function
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


if __name__ == '__main__':
    color_labels = {
        'white': 0,
        'yellow': 1,
        'blue': 2,
        'green': 3,
    }
    # transforms.Normalize((0.3738, 0.3738, 0.3738),(0.3240, 0.3240, 0.3240))
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    data_file = 'D:\\HUUCLPRD'
    train_dataset = LicensePlateDataset(data_file, "train.txt",transform=transform)
    test_dataset = LicensePlateDataset(data_file, "test.txt",transform=transform)
    val_dataset = LicensePlateDataset(data_file, "validation.txt",transform=transform)

    epoch = 20
    custom_batch_size=4
    works = 8
    prefetch=2
    train_loader = DataLoader(train_dataset, batch_size=custom_batch_size, shuffle=True,pin_memory=True,prefetch_factor=prefetch,num_workers=works)
    val_loader = DataLoader(val_dataset, batch_size=custom_batch_size, shuffle=False,pin_memory=True,prefetch_factor=prefetch,num_workers=works)
    test_loader = DataLoader(test_dataset, batch_size=custom_batch_size, shuffle=False,pin_memory=True,prefetch_factor=prefetch,num_workers=works)

    print("cuda:",torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义 ResNet 模型
    n_div = 2  # 分割因子
    model = ModifiedResNet18(n_div, len(color_labels)).to(device)# 使用预训练的 ResNet18
    custom_module = TripletAttention()
    model.resnet.conv1 = nn.Sequential(
        model.resnet.conv1,
        custom_module
    ).to(device)

    file_name = 'myResNet18_bs4_front_back_trip'
    start = time.time()
    acc,loss,_,_ = train_function.train(model=model,train_loader=train_loader,num_epochs=epoch,device=device,save_name=file_name,val_loader=val_loader)
    end = time.time()
    total_time = end - start
    print(f"训练总用时：{total_time}")
    train_function.test_model(model=model,test_loader=test_loader,device=device)
    train_function.make_acc_loss_graph(epoch=epoch,accuracies=acc,losses=loss,save_file_name=file_name)