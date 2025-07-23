import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader ,random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ===== 数据集定义 =====
class FuncRankDataset(Dataset):
    def __init__(self, data_dir):
        self.data_by_file = []
        self.scaler = StandardScaler()
        for filename in os.listdir(data_dir):
            func_list = []
            with open(os.path.join(data_dir, filename), 'r') as f:
                for line in f:
                    parts = re.split(r'[:：]', line.strip())
                    if len(parts) != 13 or 'None' in parts:
                        continue
                    try:
                        name = parts[0].strip()
                        features = [float(x.strip()) for x in parts[1:11]]
                        tag = int(parts[-1])
                        func_list.append((features, tag, name))
                    except:
                        continue
            if func_list:
                features = [f for f, _, _ in func_list]
                normalized = self.scaler.fit_transform(features)
                func_list = [(torch.tensor(f, dtype=torch.float32), tag, name)
                             for f, (_, tag, name) in zip(normalized, func_list)]
                if any(tag == 1 for _, tag, _ in func_list):
                    self.data_by_file.append(func_list)

    def __len__(self):
        return len(self.data_by_file)

    def __getitem__(self, idx):
        file_funcs = self.data_by_file[idx]
        features = torch.stack([f for f, _, _ in file_funcs])
        labels = torch.tensor([tag for _, tag, _ in file_funcs])
        names = [name for _, _, name in file_funcs]
        return features, labels, names

# ===== 模型定义 =====
class MultiTaskRankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.rank_head = nn.Linear(128, 1)
        self.class_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        score = self.rank_head(x).squeeze(-1)
        prob = torch.sigmoid(self.class_head(x).squeeze(-1))
        return score, prob

# ===== 改进后的排序+分类损失 =====
def multi_task_loss(scores, probs, labels, tail_percent=0.3, alpha=2.0, beta=1.0):
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    tail_k = int(len(neg_scores) * tail_percent)
    if tail_k == 0 or len(pos_scores) == 0:
        rank_loss = torch.tensor(0.0, requires_grad=True)
    else:
        tail_scores = torch.topk(neg_scores, tail_k, largest=False).values
        diffs = pos_scores.view(-1, 1) - tail_scores.view(1, -1)
        rank_loss = torch.mean(F.relu(1 - diffs))

    cls_loss = F.binary_cross_entropy(probs, labels.float())
    return alpha * rank_loss + beta * cls_loss, rank_loss.item(), cls_loss.item()

# ===== 验证指标 =====
def evaluate(model, dataloader, percentiles=[10,20,30]):
    model.eval()
    count = {p: 0 for p in percentiles}
    total = 0

    with torch.no_grad():
        for features, labels, _ in dataloader:
            features = features[0]
            labels = labels[0]
            scores, _ = model(features)
            _, sorted_idx = torch.sort(scores, descending=True)
            N = len(labels)
            pos_idx = (labels == 1).nonzero(as_tuple=True)[0]

            for p in percentiles:
                k = int(N * (1 - p / 100))
                tail = sorted_idx[k:]
                if any(i in tail for i in pos_idx):
                    count[p] += 1
            total += 1

    print("\n🔍 验证集中“后X%包含缺陷函数”的文件比例：")
    for p in percentiles:
        ratio = 100 * count[p] / total
        print(f"  后{p}% 区间包含缺陷函数：{ratio:.2f}%")
    return count

# ===== 训练过程 =====
def train():
    full_dataset = FuncRankDataset(data_dir="../../datasets/func_data_train")
# 计算划分大小
    train_size = int(0.82 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # 按 82/18 划分训练集和验证集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = MultiTaskRankNet(input_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_20 = float("inf")
    losses, val10s, val20s, val30s = [], [], [], []

    for epoch in range(1, 201):
        model.train()
        total_loss, total_rank, total_cls = 0, 0, 0
        for features, labels, _ in train_loader:
            features = features[0]
            labels = labels[0]
            scores, probs = model(features)
            loss, r, c = multi_task_loss(scores, probs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_rank += r
            total_cls += c

        print(f"\n📘 Epoch {epoch}")
        print(f"Loss: {total_loss:.4f} | RankLoss: {total_rank:.4f} | ClsLoss: {total_cls:.4f}")

        val_stat = evaluate(model, val_loader)
        val10s.append(val_stat[10])
        val20s.append(val_stat[20])
        val30s.append(val_stat[30])
        losses.append(total_loss)

        if val_stat[20] < best_val_20:
            best_val_20 = val_stat[20]
            torch.save(model.state_dict(), "filter_model.pt")
            print("✅ 模型已保存（后20%包含最少缺陷函数）")

    # 绘制图像
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(losses, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1,2,2)
    plt.plot(val10s, label="Tail10% Hit", marker='o')
    plt.plot(val20s, label="Tail20% Hit", marker='o')
    plt.plot(val30s, label="Tail30% Hit", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Files with Bug in Tail")
    plt.title("Validation Tail Hit Count")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_summary.png")
    plt.show()

if __name__ == "__main__":
    train()
