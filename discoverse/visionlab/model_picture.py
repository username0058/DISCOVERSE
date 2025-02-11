import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 假设输入的掩码图像大小为 (batch_size, num_views, height, width)
# 假设当前的 19 个关节动作大小为 (batch_size, 19)

class MaskEncoder(nn.Module):
    def __init__(self, num_views, hidden_dim):
        super(MaskEncoder, self).__init__()
        self.conv1 = nn.Conv2d(num_views, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_dim * 64 * 64, hidden_dim)  # 假设图像大小为 64x64

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc(x))
        return x

class ActionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ActionEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

class ActionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActionDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActionPredictionModel(nn.Module):
    def __init__(self, num_views, mask_hidden_dim, action_hidden_dim, output_dim):
        super(ActionPredictionModel, self).__init__()
        self.mask_encoder = MaskEncoder(num_views, mask_hidden_dim)
        self.action_encoder = ActionEncoder(19, action_hidden_dim)
        self.action_decoder = ActionDecoder(mask_hidden_dim + action_hidden_dim, action_hidden_dim, output_dim)

    def forward(self, mask_images, current_actions):
        mask_features = self.mask_encoder(mask_images)
        action_features = self.action_encoder(current_actions)
        combined_features = torch.cat((mask_features, action_features), dim=1)
        predicted_actions = self.action_decoder(combined_features)
        return predicted_actions

# 假设输入数据
batch_size = 32
num_views = 3
height = 64
width = 64
mask_images = torch.randn(batch_size, num_views, height, width)
current_actions = torch.randn(batch_size, 19)

# 初始化模型
model = ActionPredictionModel(num_views, mask_hidden_dim=128, action_hidden_dim=64, output_dim=19)

# 前向传播
predicted_actions = model(mask_images, current_actions)
print(predicted_actions.shape)  # 输出应为 (batch_size, 19)
