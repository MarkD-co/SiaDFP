import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from statistics import mean
import csv
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, precision_recall_curve, roc_curve, 
                             roc_auc_score, auc)
# 以下是你原有模块，保持不变
from loadData import DiskDataset
from ContrastiveLoss import ContrastiveLoss
from SiaDFPNet import SiameseNet_pos_att_cnn_adjust


def show_plot(iteration, loss, path, version, loss_type):
    plt.plot(iteration, loss)
    plt.savefig(path + loss_type + '_' + version + '.png', format='png')
    plt.show()


def mk_csv(heads, path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(heads)


def train(save_path, save_version, train_dataloader, val_dataloader, writer, thresholds, loss_type):
    train_counter = []
    train_loss_history = []
    iteration_number = 0
    val_iteration_number = 0
    val_counter = []
    val_loss_history = []
    min_val_loss = np.inf

    if not os.path.exists(save_path + save_version):
        os.makedirs(save_path + save_version)

    for epoch in tqdm(range(0, epochs)):
        train_loss = []
        val_loss = []
        net.train()  # 训练模式
        for i, data in enumerate(train_dataloader, 0):
            part1, part2, label = data
            # 移除Variable，直接使用cuda和float类型转换（PyTorch 1.0+无需Variable）
            part1 = part1.float().cuda()
            part2 = part2.float().cuda()
            label = label.float().cuda()
            
            out1, out2 = net(part1, part2)
            optimizer.zero_grad()
            loss_contrastive = criterion(out1, out2, label)
            loss_contrastive.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"Epoch number {epoch}\n Current train loss {loss_contrastive.item()}\n")
                iteration_number += 10
                train_counter.append(iteration_number)
                train_loss_history.append(loss_contrastive.item())
                train_loss.append(loss_contrastive.item())
        
        if epoch % 200 == 0:
            torch.save(net.state_dict(), f"{save_path}{save_version}/{save_version}epoch{epoch}.pt")

        writer.add_scalar('Loss/train', mean(train_loss), epoch)

        # 验证阶段
        count = {}
        label_types = ['true', 'pre']
        for threshold in thresholds:
            label_ty = {lt: [] for lt in label_types}
            count[threshold] = label_ty
        
        net.eval()  # 评估模式
        with torch.no_grad():  # 禁用梯度计算
            for i, data in enumerate(val_dataloader, 0):
                part1, part2, label = data
                part1 = part1.cuda()
                part2 = part2.cuda()
                label = label.cuda()
                
                out1, out2 = net(part1, part2)
                test_loss_contrastive = criterion(out1, out2, label)
                euclidean_distance = F.pairwise_distance(out1, out2)
                
                if loss_type == 'tanh':
                    euclidean_distance = torch.tanh(euclidean_distance)

                for threshold in thresholds:
                    count[threshold]['true'].extend(label.cpu().numpy())
                    # 复制数组避免原地修改影响后续阈值计算
                    ed_copy = euclidean_distance.clone()
                    ed_copy[ed_copy >= threshold] = 0
                    ed_copy[ed_copy < threshold] = 1
                    count[threshold]['pre'].extend(ed_copy.int().cpu().numpy())
                
                if i % 50 == 0:
                    print(f"{model}:{save_version}")
                    print(f"Epoch number {epoch}\n Current val loss {test_loss_contrastive.item()}\n")
                    val_iteration_number += 10
                    val_counter.append(val_iteration_number)
                    val_loss_history.append(test_loss_contrastive.item())
                    val_loss.append(test_loss_contrastive.item())
        
        mean_val_loss = mean(val_loss)
        writer.add_scalar('Loss/test', mean_val_loss, epoch)

        if mean_val_loss < min_val_loss:
            torch.save(net.state_dict(), f"{save_path}{save_version}/{save_version}.pt")
            min_val_loss = mean_val_loss

    writer.close()
    return net


def find_thresholds(data_loader, net):
    ps_min = 0
    ps_max = 0
    ng_min = 0
    ng_max = 0
    ps_label = torch.tensor([1.0]).cuda()
    ng_label = torch.tensor([0.0]).cuda()
    
    net.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader, 0)):
            x_0, x_1, real_label = data
            x_0 = x_0.float().cuda()
            x_1 = x_1.float().cuda()
            real_label = real_label.float().cuda()
            
            output1, output2 = net(x_0, x_1)
            euclidean_distance = F.pairwise_distance(output1, output2)
            
            if real_label.equal(ps_label):
                ps_max = max(ps_max, euclidean_distance.item())
                ps_min = min(ps_min, euclidean_distance.item())
            else:
                ng_max = max(ng_max, euclidean_distance.item())
                ng_min = min(ng_min, euclidean_distance.item())
    
    thresholds = []
    ps_min = ps_min - 1
    ng_max = ng_max + 1
    if ps_min < ng_max:
        dis = (ng_max - ps_min) / 10
        while ps_min < ng_max:
            ps_min = round(ps_min, 2)
            thresholds.append(ps_min)
            ps_min += dis
    
    print("successful!")
    if len(thresholds) == 0:
        with open('./result/output.txt', "a+") as output_file:
            output_file.write(f"{ps_min} {ng_max} {model} {save_version}\n")
    return thresholds


def plot_pr_curve(precision, recall, target_precisions, recall_at_target, model_name, save_path):
    """绘制PR曲线，标注指定precision对应的recall值"""
    plt.figure(figsize=(8, 6))
    pr_auc = auc(recall, precision)  # 计算PR曲线下面积
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})', color='blue', linewidth=2)
    
    # 标注目标precision对应的recall点
    colors = ['red', 'green', 'orange']
    for i, (target_p, recall_r) in enumerate(zip(target_precisions, recall_at_target)):
        if recall_r is not None:
            plt.scatter(recall_r, target_p, color=colors[i], s=100, 
                        label=f'Precision ≥ {target_p:.2f} → Recall = {recall_r:.4f}')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, f'PR_curve_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, model_name, save_path):
    """绘制ROC曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')  # 随机猜测基线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, f'ROC_curve_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def find_recall_at_precision(precision, recall, target_precision):
    """
    找到precision≥target_precision时的最大recall值
    :param precision: PR曲线的precision数组
    :param recall: PR曲线的recall数组
    :param target_precision: 目标precision（0.9/0.95/0.99）
    :return: 满足条件的最大recall，无则返回None
    """
    valid_indices = np.where(precision >= target_precision)[0]
    if len(valid_indices) == 0:
        return None
    return recall[valid_indices].max()


if __name__ == '__main__':
    batch_size = 4096
    test_batch_size = 4096
    lr_rate = 0.001
    epochs = 601
    drop_rate = 0.5
    hidden_size = 32
    data_year = 2017

    metric = 'Euclide'
    data_lens = [30]
    pre_lens = [10]
    models = ['cnnlstm_pos_att10_cp']
    margins = [2]
    loss_types = ['margin']
    kernel_sizes = [4]
    # 扩展结果表头，添加指定precision对应的recall列
    result_heads = [
        'model', 'name', 'threshold', 'TP', 'FP', 'TN', 'FN', 
        'Precision', 'F1', 'Accuracy', 'Recall',
        'Recall_at_90%_precision', 'Recall_at_95%_precision', 'Recall_at_99%_precision'
    ]

    result_path = f'./result/{metric}_pre_lens.csv'
    mk_csv(result_heads, result_path)

    # 定义目标precision值
    target_precisions = [0.9, 0.95, 0.99]
    # 创建曲线保存目录
    curve_save_path = './result/curves/'
    if not os.path.exists(curve_save_path):
        os.makedirs(curve_save_path)

    result_lists = []
    for model in tqdm(models):
        for data_len in tqdm(data_lens):
            for pre_len in tqdm(pre_lens):
                for kernel_size in kernel_sizes:
                    for loss_type in loss_types:
                        tanh_count = 0
                        for margin in margins:
                            tanh_count += 1
                            if loss_type == 'tanh' and tanh_count == 2:
                                continue
                            
                            data_root = f'data_{data_year}_{data_len}/supervise/data_{2017}_{data_len}_{pre_len}/'
                            file_root = './dataprocess/'
                            data_root = f'metrics/{metric}/{data_len}/{pre_len}/'
                            root = file_root + data_root
                            result_root = './checkpoint/supervise/'

                            data_version = f'{metric}_{data_len}_{pre_len}_{kernel_size}_{loss_type}'
                            if loss_type == 'margin':
                                train_version = f'_{batch_size}_{epochs}_{loss_type}{margin}'
                                thresholds = np.arange(0, margin, margin/10)
                            else:
                                train_version = f'_{batch_size}_{epochs}_{loss_type}'
                                thresholds = np.arange(0, 1, 0.1)
                            save_version = data_version + train_version
                            save_path = f'{result_root}{model}/'
                            test_model = f'{save_path}{save_version}.pt'

                            # TensorBoard配置
                            tensorboard_path = f'./runs/{model}_{save_version}'
                            if not os.path.exists(tensorboard_path):
                                os.makedirs(tensorboard_path)
                            writer = SummaryWriter(tensorboard_path)
                            numberOfSmart = {data_len: 18}

                            print('加载训练数据...')
                            train_dataset = DiskDataset()
                            train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=8, batch_size=batch_size)
                            val_dataset = DiskDataset()
                            val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, num_workers=8, batch_size=test_batch_size)

                            # 初始化模型
                            net = SiameseNet_pos_att_cnn_adjust(
                                numberOfSmart[data_len], numberOfSmart[data_len], 
                                data_len, kernel_size, model
                            ).cuda()
                            criterion = ContrastiveLoss(margin, loss_type, metric)
                            optimizer = optim.Adam(net.parameters(), lr=lr_rate)
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                            print(f'开始训练：{model}')
                            trained_model = train(save_path, save_version, train_dataloader, val_dataloader, writer, thresholds, loss_type)
                            print("模型训练完成并保存")

                            # 测试阶段
                            print("加载测试模型...")
                            test_net = SiameseNet_pos_att_cnn_adjust(
                                numberOfSmart[data_len], numberOfSmart[data_len], 
                                data_len, kernel_size, model
                            ).cuda()
                            print("加载测试数据...")
                            test_data = DiskDataset(False)
                            test_dataloader = DataLoader(dataset=test_data, shuffle=True, batch_size=2048, num_workers=8)

                            # 遍历保存的模型文件
                            model_dir = f'{save_path}{save_version}/'
                            if not os.path.exists(model_dir):
                                continue
                            
                            for model_name in tqdm(os.listdir(model_dir)):
                                if '.pt' not in model_name:
                                    continue

                                # 加载模型
                                test_model_path = f'{model_dir}{model_name}'
                                test_net.load_state_dict(torch.load(test_model_path, map_location=device))
                                test_net.eval()  # 设置为评估模式
                                print(f"成功加载测试模型：{model_name}")

                                # 1. 收集所有真实标签和预测概率（用于PR/ROC曲线）
                                all_real_labels = []
                                all_pred_probs = []
                                max_distance = 0.0  # 用于归一化距离到0-1区间

                                with torch.no_grad():  # 禁用梯度计算，加速测试
                                    for i, data in tqdm(enumerate(test_dataloader), desc="收集测试数据"):
                                        x_0, x_1, real_label = data
                                        x_0 = x_0.float().to(device)
                                        x_1 = x_1.float().to(device)
                                        real_label = real_label.float().to(device)

                                        output1, output2 = test_net(x_0, x_1)
                                        euclidean_distance = F.pairwise_distance(output1, output2)

                                        # 处理tanh归一化
                                        if loss_type == 'tanh':
                                            euclidean_distance = torch.tanh(euclidean_distance)

                                        # 更新最大距离（用于概率归一化）
                                        current_max = euclidean_distance.max().item()
                                        if current_max > max_distance:
                                            max_distance = current_max

                                        # 收集真实标签
                                        all_real_labels.extend(real_label.cpu().numpy())
                                        # 计算正类概率：距离越小，相似性越高，概率越大（1 - 归一化距离）
                                        if max_distance > 0:
                                            pred_prob = 1 - (euclidean_distance / max_distance)
                                        else:
                                            pred_prob = torch.ones_like(euclidean_distance)
                                        all_pred_probs.extend(pred_prob.cpu().numpy())

                                # 转换为numpy数组
                                all_real_labels = np.array(all_real_labels).astype(int)
                                all_pred_probs = np.array(all_pred_probs)

                                # 2. 计算PR曲线并找到指定precision对应的recall
                                precision, recall, _ = precision_recall_curve(all_real_labels, all_pred_probs, pos_label=1)
                                recall_at_target = [find_recall_at_precision(precision, recall, tp) for tp in target_precisions]

                                # 3. 计算ROC曲线和AUC
                                fpr, tpr, _ = roc_curve(all_real_labels, all_pred_probs, pos_label=1)
                                roc_auc = roc_auc_score(all_real_labels, all_pred_probs)

                                # 4. 绘制并保存PR/ROC曲线
                                plot_pr_curve(precision, recall, target_precisions, recall_at_target, model_name, curve_save_path)
                                plot_roc_curve(fpr, tpr, roc_auc, model_name, curve_save_path)

                                # 5. 输出指定precision对应的recall结果
                                print(f"\n模型 {model_name} 关键结果：")
                                for tp, tr in zip(target_precisions, recall_at_target):
                                    status = f"{tr:.4f}" if tr is not None else "无满足条件的值"
                                    print(f"Precision ≥ {tp*100:.0f}% 时，Recall = {status}")

                                # 6. 保留原有阈值循环逻辑，补充新指标到CSV
                                result_lists = []
                                for threshold in tqdm(thresholds, desc="计算不同阈值指标"):
                                    real_labels = []
                                    pre_labels = []
                                    with torch.no_grad():
                                        for i, data in enumerate(test_dataloader):
                                            x_0, x_1, real_label = data
                                            x_0 = x_0.float().to(device)
                                            x_1 = x_1.float().to(device)
                                            real_label = real_label.float().to(device)

                                            output1, output2 = test_net(x_0, x_1)
                                            euclidean_distance = F.pairwise_distance(output1, output2)

                                            if loss_type == 'tanh':
                                                euclidean_distance = torch.tanh(euclidean_distance)

                                            # 生成预测标签
                                            pred_label = (euclidean_distance < threshold).int()
                                            real_labels.extend(real_label.cpu().numpy())
                                            pre_labels.extend(pred_label.cpu().numpy())

                                    # 转换标签类型
                                    real_labels = [int(x) for x in real_labels]
                                    pre_labels = [int(x) for x in pre_labels]

                                    # 计算传统指标（修正pos_label为1，符合正类定义）
                                    confusion_ = confusion_matrix(real_labels, pre_labels)
                                    TN, FP, FN, TP = confusion_.ravel()
                                    precision_ = precision_score(real_labels, pre_labels, pos_label=1, zero_division=0)
                                    accuracy_ = accuracy_score(real_labels, pre_labels)
                                    f1_ = f1_score(real_labels, pre_labels, pos_label=1, zero_division=0)
                                    recall_ = recall_score(real_labels, pre_labels, pos_label=1, zero_division=0)

                                    # 构建结果行（补充新的recall_at_target列）
                                    result_list = [
                                        model, model_name, threshold, TP, FP, TN, FN,
                                        precision_, f1_, accuracy_, recall_,
                                        recall_at_target[0] if recall_at_target[0] is not None else '',
                                        recall_at_target[1] if recall_at_target[1] is not None else '',
                                        recall_at_target[2] if recall_at_target[2] is not None else ''
                                    ]
                                    result_lists.append(result_list)

                                # 7. 保存结果到CSV
                                with open(result_path, 'a+', newline='') as file:
                                    result_file = csv.writer(file)
                                    result_file.writerows(result_lists)
                                print(f"结果已保存到 {result_path}，曲线已保存到 {curve_save_path}")

    print("所有任务完成！")