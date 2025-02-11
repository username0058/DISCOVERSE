import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 生成一些随机数据
X_train = np.random.rand(1000, 10).astype(np.float32)  # 1000个样本，每个样本有10个特征
y_train = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)  # 二分类任务

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层：输入到隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 第二层：隐藏层到隐藏层
        self.fc3 = nn.Linear(hidden_size, output_size)  # 第三层：隐藏层到输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 经过第一层和ReLU激活
        x = self.relu(self.fc2(x))  # 经过第二层和ReLU激活
        x = self.fc3(x)  # 经过输出层
        return x

# 初始化模型，损失函数和优化器
model = MLP(input_size=10, hidden_size=64, output_size=1)  # 输入10个特征，隐藏层64个神经元，输出1个值
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # 清零梯度

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    outputs = model(X_train)
    predictions = torch.sigmoid(outputs)  # 输出概率
    predicted_labels = (predictions > 0.5).float()  # 将概率转换为0或1的标签
    accuracy = (predicted_labels == y_train).float().mean()
    print(f'Accuracy: {accuracy:.4f}')
