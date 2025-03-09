import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def uniform_sample_contour(contour_pts: np.ndarray, target_num: int) -> np.ndarray:
        """
        轮廓点均匀采样函数
        功能：
        - 自动处理空轮廓（返回全零数组）
        - 智能循环填充不足点数
        - 保留轮廓的几何周期性特征
        
        参数：
        contour_pts: 原始轮廓点数组，形状(N,2)
        target_num: 目标采样点数
        
        返回：
        采样后的轮廓点数组，形状(target_num,2)
        """
        if len(contour_pts) == 0:
            return np.zeros((target_num, 2), dtype=np.float32)
        
        # 计算采样间隔（浮点数精度）
        step = max(1.0, len(contour_pts) / target_num)
        
        # 生成采样索引（带循环保护）-- 如果点数不够就会来回循环
        indices = (np.arange(target_num) * step) % len(contour_pts)
        indices = indices.astype(int)
        
        # 执行采样
        sampled = contour_pts[indices]
        
        # 类型转换保证输出一致性
        return sampled.astype(np.float32)

class MaskMotionPredictor:
    """
    掩膜-运动预测系统
    功能：
    1. 解析多目标分割掩膜
    2. 提取物体轮廓及质心
    3. 生成几何特征向量
    4. 预测机械臂运动方向
    
    输入：
    - 目标物体名称 (target_block列表中的元素)
    - 灰度掩膜图 (0-255)
    
    输出：
    - 机械臂运动方向向量 (dx, dy)
    """
    
    def __init__(self, 
                 target_blocks: List[str],
                 input_dim: int):
        # 初始化目标物体映射表
        self.sample_count = 200 # 轮廓点采样数
        self.target_blocks = target_blocks
        self.gray_mapping = self._create_gray_mapping()
        
        # 初始化神经网络
        self.model = self._build_model(input_dim)
        
        # 特征缓存器
        self.feature_cache = {}
    
    def _create_gray_mapping(self) -> Dict[str, int]:
        """创建物体名称到灰度值的映射字典"""
        return {
            obj: (i + 1) * 255 // len(self.target_blocks)
            for i, obj in enumerate(self.target_blocks)
        }
    
    def _build_model(self, input_dim: int) -> nn.Module:
        """构建PyTorch动态网络结构"""
        # 以下为不加dropout与resnet的网络
        # class MotionMLP(nn.Module):
        #     def __init__(self, in_dim):
        #         super().__init__()
        #         self.input_dim = in_dim
        #         self.layers = nn.Sequential(
        #             nn.Linear(in_dim, in_dim * 2),
        #             nn.ReLU(),
        #             nn.LayerNorm(in_dim * 2),
        #             nn.Linear(in_dim * 2, in_dim * 4),
        #             nn.ReLU(),
        #             nn.LayerNorm(in_dim * 4),
        #             nn.Linear(in_dim * 4, in_dim * 2),
        #             nn.ReLU(),
        #             nn.LayerNorm(in_dim * 2),
        #             nn.Linear(in_dim * 2, 2)  #输出x,y方向
        #         )
            
        #     def forward(self, x):
        #         return torch.sigmoid(self.layers(x)) * 2 - 1  # 输出归一化到[-1, 1]
        # 以下为添加dropout与resnet的网络
        class MotionMLP(nn.Module):
            def __init__(self, in_dim):
                super().__init__()
                self.input_dim = in_dim
                
                # 主干网络
                self.trunk = nn.Sequential(
                    self._make_block(in_dim, in_dim*2),
                    self._make_block(in_dim*2, in_dim*4),
                    self._make_block(in_dim*4, in_dim*8),
                    self._make_block(in_dim*8, in_dim*4),
                    self._make_block(in_dim*4, in_dim*2),
                    nn.Linear(in_dim*2, 2)
                )
                
                # 初始化参数
                self._init_weights()
                
            def _make_block(self, in_d, out_d):
                """构建带有残差连接的块"""
                return nn.Sequential(
                    nn.Linear(in_d, out_d),
                    nn.LayerNorm(out_d),
                    nn.ReLU(),
                    nn.Dropout(0.3)# 取dropout层为0.3防止过拟合
                )
            # 初始化策略简介：
            # Xavier初始化，也被称为Glorot初始化（以其发明者Xavier Glorot命名），是一种针对深度神经网络的权重初始化方法。这种方法的主要目标是在网络训练开始时，保持每一层输入和输出的方差大致相等。

            # 以下是Xavier初始化的关键点：

            # 目的：

            # 防止深层网络中的梯度消失或爆炸问题。
            # 使得信号（激活值和梯度）能够在网络中平稳地向前和向后传播。
            # 原理：

            # 初始化权重时，考虑了每一层的输入和输出神经元的数量。
            # 权重从一个均值为0，方差为2/(nin + nout)的分布中随机采样，其中nin是输入神经元的数量，nout是输出神经元的数量。
            # 实现方式：

            # 对于均匀分布：权重在[-limit, limit]范围内均匀采样，其中limit = sqrt(6 / (nin + nout))
            # 对于正态分布：权重从均值为0，标准差为sqrt(2 / (nin + nout))的正态分布中采样
            # 适用性：

            # 特别适用于使用tanh或sigmoid等饱和激活函数的网络。
            # 对于ReLU激活函数，通常推荐使用He初始化（Xavier的一个变体）。
            # 优势：

            # 有助于保持梯度在反向传播过程中不会太小（避免梯度消失）或太大（避免梯度爆炸）。
            # 可以加速网络的收敛。
            def _init_weights(self):
                """Xavier初始化"""
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight) # 使用正态分布
                        # nn.init.xavier_uniform_(tensor)  # 使用均匀分布
                        nn.init.constant_(m.bias, 0.1)
                        
            def forward(self, x):
                # 残差连接实现
                x = self.trunk[0](x) + x  # 第一层残差
                x = self.trunk[1](x)
                x = self.trunk[2](x) + x  # 第三层残差
                x = self.trunk[3](x)
                x = self.trunk[4](x) + x  # 第五层残差
                return torch.tanh(self.trunk[5](x))  # 输出使用tanh归一化到[-1,1]
     
        return MotionMLP(input_dim)
    
    def _extract_features(self, mask: np.ndarray, target: str) -> np.ndarray:
        """
        核心特征提取流水线
        步骤：
        1. 二值化目标物体掩膜
        2. 轮廓检测（保留所有点）
        3. 质心计算
        4. 生成特征向量：归一化边缘坐标 + 归一化质心坐标
        """
        # 获取目标灰度值
        target_gray = self.gray_mapping[target]
        
        # 二值化处理 (±3灰度容差)
        _, binary = cv2.threshold(
            mask, 
            target_gray - 3, 
            target_gray + 3, 
            cv2.THRESH_BINARY
        )
        
        # 轮廓检测（保留所有点）
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        # 选取最大轮廓
        if len(contours) == 0:
            return np.zeros(self.model.input_dim)
        main_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓质心
        M = cv2.moments(main_contour)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        
        
        # 处理轮廓点坐标
        contour_pts = main_contour.squeeze(axis=1).astype(np.float32)
        sampled_contour_pts = uniform_sample_contour(contour_pts, self.sample_count)
        # 坐标归一化处理
        height, width = mask.shape[:2]
        normalized_centroid = [cx / width, cy / height]
        normalized_contour = sampled_contour_pts / np.array([width, height])
        
        # 特征拼接
        feature = np.concatenate([
            normalized_contour.flatten(),
            normalized_centroid
        ]) # 200 + 1 = 201维  （201*2）
        
        # 缓存特征用于调试
        self.feature_cache = {
            "contour": main_contour,
            "centroid": (cx, cy),
            "feature_vector": feature
        }
        
        return feature
        
    def predict_motion(self, mask: np.ndarray, target: str) -> Tuple[float, float]:
        """
        完整预测流程
        输入：
        - mask: 灰度掩膜图 (H, W) np.ndarray
        - target: 目标物体名称
        
        输出：
        - dx: x方向运动量 (-1 ~ 1)
        - dy: y方向运动量 (-1 ~ 1)
        """
        # 特征提取
        feature = self._extract_features(mask, target)
        
        # 转换为Tensor
        input_tensor = torch.FloatTensor(feature).unsqueeze(0)
        
        # 网络推理
        with torch.no_grad():
            output = self.model(input_tensor).squeeze().numpy()
        
        return output[0], output[1]
    
    def visualize(self, mask: np.ndarray) -> np.ndarray:
        """
        基础可视化：原始坐标空间
        显示内容：
        - 绿色：物体轮廓
        - 红色：质心位置
        """
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if "contour" in self.feature_cache:
            cv2.drawContours(display, [self.feature_cache["contour"]], -1, (0,255,0), 2)
            cv2.circle(display, 
                      self.feature_cache["centroid"], 
                      5, (0,0,255), -1)
        return display
    
    def visualize_normalization(self, mask: np.ndarray) -> np.ndarray:
        """
        归一化效果可视化（512x512标准空间）
        显示内容：
        - 绿色：归一化后的轮廓点
        - 红色：归一化后的质心位置
        """
        display = np.zeros((512, 512, 3), dtype=np.uint8)
        
        if "contour" in self.feature_cache:
            # 绘制归一化轮廓点
            raw_contour = self.feature_cache["contour"].squeeze(axis=1)
            for pt in raw_contour:
                x = int(pt[0] / mask.shape[1] * 512)
                y = int(pt[1] / mask.shape[0] * 512)
                cv2.circle(display, (x, y), 2, (0,255,0), -1)
            
            # 绘制归一化质心
            cx = int(self.feature_cache["centroid"][0] / mask.shape[1] * 512)
            cy = int(self.feature_cache["centroid"][1] / mask.shape[0] * 512)
            cv2.circle(display, (cx, cy), 5, (0,0,255), -1)
        
        return display
    def train(self, 
             train_dir: str,
             val_dir: str,
             epochs: int = 1000,
             batch_size: int = 32,
             lr: float = 1e-6, 
             patience: int = 30): # 早停机制
            """
            完整训练流程
            参数：
            - train_dir: 训练集目录
            - val_dir: 验证集目录
            - epochs: 训练轮次
            - batch_size: 批大小
            - lr: 学习率
            """
            # 初始化数据集
            train_dataset = MaskDataset(train_dir, self)
            val_dataset = MaskDataset(val_dir, self)
            
            # 数据加载器 训练数据被打乱，而验证数据没有
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=self._collate_fn)
            val_loader = DataLoader(val_dataset, batch_size, collate_fn=self._collate_fn)
            
            # 优化器和损失函数
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            # 学习率调度 使用ReduceLROnPlateau学习率调度器，当验证集上的损失不再下降时，学习率将乘以0.1
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            criterion = nn.MSELoss()
            # 初始化早停变量
            best_val_loss = float('inf')
            no_improve_epochs = 0
            # 训练循环
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                
                # 训练阶段
                for features, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪，确保梯度的大小保持在一个合理的范围内
                    # 计算所有模型参数梯度的L2范数（欧几里得范数）。
                    # 如果这个范数超过了指定的最大值（在这里是1.0），那么所有梯度会被等比例缩小，使得它们的范数等于这个最大值
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证阶段
                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for features, labels in val_loader:
                        outputs = self.model(features)
                        val_loss += criterion(outputs, labels).item()
                # 学习率调整
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                # 早停机制
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve_epochs = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
                # 打印训练进度
                avg_train = train_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {avg_train:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}")
    def _collate_fn(self, batch):
            """自定义批处理函数，将批次中的样本堆叠成张量"""
            features, labels = zip(*batch)
            return torch.stack(features), torch.stack(labels)
class MaskDataset(Dataset):
    """
    掩膜-运动数据集
    目录结构：
    dataset/
    ├── train/
    │   ├── images/  # 存放掩膜图
    │   └── labels/  # 存放CSV文件（dx, dy）
    └── val/
        ├── images/
        └── labels/
    """
    
    def __init__(self, 
                 root_dir: str, 
                 predictor: MaskMotionPredictor,  # 传入预测器实例 
                 img_size=(512, 512)):
        self.root = Path(root_dir)
        self.image_files = sorted((self.root / "images").glob("*.png"))
        self.label_files = sorted((self.root / "labels").glob("*.csv"))
        self.predictor = predictor
        self.target_blocks = predictor.target_blocks
        self.img_size = img_size
        self.feature_dim = predictor.model.input_dim
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载掩膜图
        mask = cv2.imread(str(self.image_files[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)
        
        # 加载标签（CSV格式：target_name, dx, dy）
        label = np.loadtxt(self.label_files[idx], delimiter=',', dtype=str)
        target_name = label[0]
        dx, dy = map(float, label[1:])
        
        # 转换为特征向量
        with torch.no_grad():
            feature = self.predictor._extract_features(mask, target_name)
        # 维度对齐
        feature = self._align_feature_dim(feature)
        return torch.FloatTensor(feature), torch.FloatTensor([dx, dy])
    def _align_feature_dim(self, feature: np.ndarray) -> np.ndarray:
        """保证特征维度一致性"""
        if len(feature) < self.feature_dim:
            #如果特征向量太短：用零填充特征向量，使其达到所需的维度。
            return np.pad(feature, (0, self.feature_dim - len(feature)))
        return feature[:self.feature_dim] # 返回特征向量的一个切片，只取前 self.feature_dim 个元素

# 推理示例
if __name__ == "__main__":
    target_block = ["bridge1","bridge2","block1_green","block2_green",
                   "block_purple1","block_purple2","block_purple3",
                   "block_purple4","block_purple5","block_purple6"]
    
    # 初始化预测器
    predictor = MaskMotionPredictor(target_block,input_dim=201)# 特征维度201假设
    predictor.model.load_state_dict(torch.load('best_model.pth'))
    predictor.model.eval()# 切换到推理模式，关闭dropout等训练专用层
    # 模拟输入 (512x512掩膜)
    dummy_mask = np.zeros((512,512), dtype=np.uint8)
    cv2.circle(dummy_mask, (256,256), 50, 25, -1)  # 模拟目标物体
    
    # 执行预测
    dx, dy = predictor.predict_motion(dummy_mask, "block1_green")
    print(f"Movement vector: ({dx:.3f}, {dy:.3f})")
    
    # 可视化检查
    debug_img = predictor.visualize(dummy_mask)
    norm_img = predictor.visualize_normalization(dummy_mask)
    
    cv2.imshow("Original Space", debug_img)
    cv2.imshow("Normalized Space", norm_img)
    cv2.waitKey(0)

'''
# 训练示例
if __name__ == "__main__":
    target_block = ["bridge1","bridge2","block1_green","block2_green",
                   "block_purple1","block_purple2","block_purple3",
                   "block_purple4","block_purple5","block_purple6"]
    
    # 初始化预测器
    predictor = MaskMotionPredictor(target_blocks, input_dim=201)
    
    # 训练配置
    predictor.train(
        train_dir="dataset/train",
        val_dir="dataset/val",
        epochs=500,
        batch_size=64,
        lr=1e-4,
        patience=15
    )
    
    # 保存/加载模型
    torch.save(predictor.model.state_dict(), "trained_model.pth")
'''