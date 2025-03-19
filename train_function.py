import math
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import time
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import csv
import seaborn as sns
color_labels = {
    'white': 0,
    'yellow': 1,
    'blue': 2,
    'green': 3,
}




def train_var_lr(model, train_loader, val_loader, num_epochs, device, save_name):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 创建CSV文件并写入表头
    with open(f'{save_name}_metrics.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Train Accuracy', 'Train Loss', 'Validate Accuracy', 'Validate Loss', 'Learning Rate'])

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  
        
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, colors) in loop:
            # 将数据移动到正确的设备上
            images = images.to(device)
            labels = torch.tensor([color_labels[color] for color in colors]).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算当前batch的平均损失和准确率
            avg_loss = epoch_loss / (batch_idx + 1)
            accuracy = 100 * correct / total

            # 更新tqdm进度条描述
            loop.set_postfix(loss=avg_loss, accuracy=accuracy)

        # 计算每个epoch的平均损失和准确率
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # 验证
        val_accuracy, val_loss, _, _, _, _ = verify(model, val_loader, device, f'{save_name}_epoch_{epoch+1}')
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 将指标写入CSV文件
        with open(f'{save_name}_metrics.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch+1, train_accuracy, avg_train_loss, val_accuracy, val_loss, current_lr])

        # 计算并打印 epoch 用时
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds. ")
        print(f"Train Accuracy: {train_accuracy:.2f}%, Train Loss: {avg_train_loss:.4f}")
        print(f"Validate Accuracy: {val_accuracy:.2f}%, Validate Loss: {val_loss:.4f}")
        print(f"Current Learning Rate: {current_lr:.6f}")

    torch.save(model.state_dict(), f'{save_name}.pth') 
    print(f"模型已保存为 '{save_name}.pth'")
    return train_accuracies, train_losses, val_accuracies, val_losses



def train(model, train_loader, val_loader, num_epochs, device, save_name):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # 创建CSV文件并写入表头
    with open(f'{save_name}_metrics.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Train Accuracy', 'Train Loss', 'Validate Accuracy', 'Validate Loss'])

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  
        
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, colors) in loop:
            # 将数据移动到正确的设备上
            images = images.to(device)
            labels = torch.tensor([color_labels[color] for color in colors]).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算当前batch的平均损失和准确率
            avg_loss = epoch_loss / (batch_idx + 1)
            accuracy = 100 * correct / total

            # 更新tqdm进度条描述
            loop.set_postfix(loss=avg_loss, accuracy=accuracy)

        # 计算每个epoch的平均损失和准确率
        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # 验证
        val_accuracy, val_loss, _, _, _, _ = verify(model, val_loader, device, f'{save_name}_epoch_{epoch+1}')
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 将指标写入CSV文件
        with open(f'{save_name}_metrics.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch+1, train_accuracy, avg_train_loss, val_accuracy, val_loss])

        # 计算并打印 epoch 用时
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds. ")
        print(f"Train Accuracy: {train_accuracy:.2f}%, Train Loss: {avg_train_loss:.4f}")
        print(f"Validate Accuracy: {val_accuracy:.2f}%, Validate Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), f'{save_name}.pth') 
    print(f"模型已保存为 '{save_name}.pth'")
    return train_accuracies, train_losses, val_accuracies, val_losses

def verify(model, val_loader, device, write_file_name):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        all_labels = []
        all_predictions = []
        correct_predictions = []
        incorrect_predictions = []
        batch_accuracies=[]
        correct = 0
        total = 0
        val_loss = 0.0

        # 添加 tqdm 进度条
        loop = tqdm(val_loader, total=len(val_loader), desc="验证中")
        for images, colors in loop:
            images = images.to(device)
            labels = torch.tensor([color_labels[color] for color in colors]).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 计算当前批次的准确率
            batch_accuracy = (predicted == labels).sum().item() / labels.size(0)
            
            # 更新进度条描述
            loop.set_postfix(loss=loss.item(), accuracy=100*batch_accuracy)

            # 记录每个样本的预测结果
            for i in range(labels.size(0)):
                true_label = colors[i]
                predicted_label = list(color_labels.keys())[predicted[i]]
                if predicted[i] == labels[i]:
                    correct_predictions.append(f"真实车牌: {true_label}, 预测结果: {predicted_label}\n")
                else:
                    incorrect_predictions.append(f"真实车牌: {true_label}, 预测结果: {predicted_label}\n")

        # 计算平均准确率和损失
        average_accuracy = 100 * correct / total
        average_loss = val_loss / len(val_loader)

        # 计算评估指标
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # 计算特异度
        cm = confusion_matrix(all_labels, all_predictions)
        specificity = np.sum(np.diag(cm)) / np.sum(cm)

        print(f'Validate Accuracy: {average_accuracy:.2f}%')
        print(f'Validate Loss: {average_loss:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall (Sensitivity): {recall:.4f}')
        print(f'F1-score: {f1:.4f}')
        print(f'Specificity: {specificity:.4f}')

        # 将结果写入文件
        # with open(f'correct_{write_file_name}.txt', 'a', encoding='utf-8') as correct_file:
        #     correct_file.writelines(correct_predictions)

        # with open(f'incorrect_{write_file_name}.txt', 'a', encoding='utf-8') as incorrect_file:
        #     incorrect_file.writelines(incorrect_predictions)

        # print(f"预测结果已写入文件: correct_{write_file_name}.txt 和 incorrect_{write_file_name}.txt")

        return average_accuracy, average_loss, precision, recall, f1, specificity
    
def test_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        loop = tqdm(test_loader, total=len(test_loader), desc="测试中")
        for images, colors in loop:
            images = images.to(device)
            labels = torch.tensor([color_labels[color] for color in colors]).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # 更新进度条
            loop.set_postfix(loss=loss.item(), accuracy=100*correct/total)

    # 计算平均损失和准确率
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    # 计算其他评估指标
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # 计算特异度
    cm = confusion_matrix(all_labels, all_predictions)
    specificity = np.sum(np.diag(cm)) / np.sum(cm)

    # 打印结果
    print(f"\n测试结果:")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return accuracy, avg_loss, precision, recall, f1, specificity

def make_acc_loss_graph(epoch,accuracies,losses,save_file_name):
    x1 = range(0, epoch+1)
    x2 = range(0, epoch+1)
    y1 = accuracies
    y2 = losses
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig(f"{save_file_name}.jpg")


def lr_search(model, train_loader, val_loader, device, num_epochs, save_name):
    # 定义要尝试的学习率
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    best_lr = None
    best_val_accuracy = 0

    for lr in learning_rates:
        print(f"\n尝试学习率: {lr}")
        
        # 重置模型参数
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        
        optimizer = Adam(model.parameters(), lr=lr)
        
        # 训练模型
        train_accuracies, train_losses, val_accuracies, val_losses = train(
            model, train_loader, val_loader, num_epochs, device, f"{save_name}_lr_{lr}"
        )
        
        # 获取最后一个 epoch 的验证准确率
        final_val_accuracy = val_accuracies[-1]
        
        print(f"学习率 {lr} 的最终验证准确率: {final_val_accuracy:.2f}%")
        
        # 更新最佳学习率
        if final_val_accuracy > best_val_accuracy:
            best_val_accuracy = final_val_accuracy
            best_lr = lr

    print(f"\n最佳学习率: {best_lr}, 验证准确率: {best_val_accuracy:.2f}%")
    return best_lr



def find_lr(model, train_loader, optimizer, criterion, device, init_value=1e-8, final_value=10., beta=0.98):
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    
    model.train()
    for images, colors in train_loader:
        batch_num += 1
        
        images = images.to(device)
        labels = torch.tensor([color_labels[color] for color in colors]).to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 计算平滑损失
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        
        # 如果损失爆炸，停止
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        
        # 记录最佳损失
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        
        # 存储值
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新下一步的学习率
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    
    return log_lrs, losses

# 使用函数并绘制结果
def plot_lr_finder(log_lrs, losses):
    plt.plot(log_lrs[10:-5], losses[10:-5])
    plt.xscale('log')
    plt.xlabel('Learning rate')
    plt.ylabel('Loss')
    plt.savefig('lr_finder_plot.png')
    plt.show()
    plt.close()

