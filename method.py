import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchvision.models as models
import vit
import cv2
import pywt
import numpy as np
import pandas as pd
import Resnet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from skimage.segmentation import slic
from skimage.util import img_as_float
import torch.nn.functional as F
from collections import defaultdict

# 读取CSV文件
dataframe = pd.read_csv('/home/user/nihao/juhua_19.csv', encoding='GBK')


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        初始化数据集
        :param dataframe: 包含正反面图像路径和标签的DataFrame
        :param transform: 图像的预处理方法
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        front_img_path = self.dataframe.iloc[idx, 0]
        back_img_path = self.dataframe.iloc[idx, 1]
        label = self.dataframe.iloc[idx, 2]

        try:
            front_img = Image.open(front_img_path).convert('RGB')
            back_img = Image.open(back_img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 出错图像：\n  正面: {front_img_path}\n  反面: {back_img_path}\n  错误信息: {e}")
            # 返回一张纯黑图像代替损坏图片
            front_img = Image.new('RGB', (224, 224))
            back_img = Image.new('RGB', (224, 224))

        if self.transform:
            front_img = self.transform(front_img)
            back_img = self.transform(back_img)

        return front_img, back_img, label


    def load_and_preprocess_image_HSV(self, image):
        # Convert image from PIL format to numpy array
        image_np = np.array(image)
        # Convert image from RGB to HSV
        image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        return image_hsv

    def load_and_preprocess_image_LAB(self, image):
        # Convert image from PIL format to numpy array
        image_np = np.array(image)
        # Convert image from RGB to LAB
        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        return image_lab


# 定义训练数据预处理转换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为相同的尺寸
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义测试数据预处理转换
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为相同的尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建自定义数据集
train_dataset = CustomDataset(dataframe=dataframe, transform=train_transform)
test_dataset = CustomDataset(dataframe=dataframe, transform=test_transform)

# 划分数据集，80% 训练集，20% 测试集
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

pretrained_cfg_overlay = {'file': r"/home/user/nihao/pytorch_model.bin"}
vit_model = timm.create_model('vit_base_patch16_224', pretrained_cfg_overlay=pretrained_cfg_overlay,
                              pretrained=True)
VIT = vit.VisionTransformer()
state_dict = vit_model.state_dict()
VIT.load_state_dict(state_dict)

resnet1 = models.resnet50(pretrained = True)
net_re1 = Resnet.resnet50(pretrained=False)
state_dict1 = resnet1.state_dict()
net_re1.load_state_dict(state_dict1)

resnet2 = models.resnet50(pretrained = True)
net_re2 = Resnet.resnet50(pretrained=False)
state_dict2 = resnet2.state_dict()
net_re2.load_state_dict(state_dict2)

def wavelet_decomposition(image):
    # 将图像移动到CPU上并获取不需要梯度的张量
    image_cpu = image.cpu().detach()

    # 小波分解
    LL, (LH, HL, HH) = pywt.dwt2(image_cpu.numpy(), 'haar')

    # 将结果移动到GPU上
    LL = torch.from_numpy(LL).to(image.device)
    LH = torch.from_numpy(LH).to(image.device)

    return LL, LH

# 定义自定义的ResNet_18模型
class ResNet_18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_18, self).__init__()

        # 加载预训练的ResNet-18模型
        self.resnet1 = net_re1
        self.resnet2 = net_re2
        
        self.maxpool3 = nn.MaxPool2d(4)
        self.maxpool4 = nn.MaxPool2d(8)
        self.maxpool5 = nn.MaxPool2d(7)
        self.maxpool9 = nn.MaxPool2d(2)
        
        self.maxpool6 = nn.MaxPool2d(4)
        self.maxpool7 = nn.MaxPool2d(8)
        self.maxpool8 = nn.MaxPool2d(7)
        self.maxpool10 = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(256, 768, 3, 1, 1)
        self.conv2 = nn.Conv2d(512, 768, 3, 1, 1)
        self.conv3 = nn.Conv2d(1024, 768, 3, 1, 1)
        
        self.conv5 = nn.Conv2d(256, 768, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 768, 3, 1, 1)
        self.conv7 = nn.Conv2d(1024, 768, 3, 1, 1)
        
        self.conv8 = nn.Conv1d(2304, 1500, 3, 1, 1)
        self.conv9 = nn.Conv1d(2304, 1500, 3, 1, 1)
        
        self.conv4 = nn.Conv2d(3072, 1024, 3, 1, 1)
        
        self.vit1 = VIT
        self.vit2 = VIT
        self.vit3 = VIT

        self.fc1 = nn.Linear(1500, 500)
        self.fc2 = nn.Linear(1500, 500)
        
        self.fc3 = nn.Linear(1000,num_classes)

    def forward(self, x1, x2):
    
        feature1_rgb, feature2_rgb, feature3_rgb = self.resnet1(x1)
        feature4_rgb, feature5_rgb, feature6_rgb = self.resnet2(x2)
        
        feature1_rgb_LL = self.maxpool4(feature1_rgb)
        feature2_rgb_LL = self.maxpool3(feature2_rgb)
        feature3_rgb = self.maxpool9(feature3_rgb)
        
        
        feature4_rgb_LL = self.maxpool7(feature4_rgb)
        feature5_rgb_LL = self.maxpool6(feature5_rgb)
        feature6_rgb = self.maxpool10(feature6_rgb)
        
        
        out1 = self.conv1(feature1_rgb_LL)
        out2 = self.conv2(feature2_rgb_LL)
        out3 = self.conv3(feature3_rgb)
        
        out4 = self.conv5(feature4_rgb_LL)
        out5 = self.conv6(feature5_rgb_LL)
        out6 = self.conv7(feature6_rgb)
        
        out1_1 = torch.concat([out1,out2,out3], dim=1)
        out1_2 = torch.concat([out4,out5,out6], dim=1)
        
        out1_1 = self.maxpool5(out1_1)
        out1_2 = self.maxpool8(out1_2)
        
        out1_1 = out1_1.view(out1_1.size(0), -1, 1)
        out1_2 = out1_2.view(out1_2.size(0), -1, 1)
        
        out1_1 = self.conv8(out1_1)
        out1_2 = self.conv9(out1_2)
        
        out1_1 = out1_1.view(out1_1.size(0), -1)
        out1_2 = out1_2.view(out1_2.size(0), -1)

        out1_1 = self.fc1(out1_1)
        out1_2 = self.fc2(out1_2)
        
        out = torch.concat([out1_1,out1_2], dim=1)
        
        out = self.fc3(out)
        
        return out


# 定义模型 
model = ResNet_18(num_classes=19)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for image1,image2, labels in tqdm(train_loader):
        image1,image2, labels = image1.to(device),image2.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(image1,image2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # 评估模型在验证集上的表现
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for image1,image2, labels in test_loader:
            image1,image2, labels = image1.to(device),image2.to(device), labels.to(device)
            outputs = model(image1,image2)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# 类别统计：每类预测正确数与总样本数
class_correct = defaultdict(int)
class_total = defaultdict(int)

with torch.no_grad():
    for image1, image2, labels in test_loader:
        image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)
        outputs = model(image1, image2)
        _, predicted = torch.max(outputs, 1)

        for label, pred in zip(labels, predicted):
            class_total[int(label)] += 1
            if label == pred:
                class_correct[int(label)] += 1

# 打印每个类别的准确率
print("\nPer-class accuracy:")
for cls in sorted(class_total.keys()):
    acc = 100 * class_correct[cls] / class_total[cls]
    print(f"Class {cls}: {acc:.2f}%")
    
# 保存模型参数
torch.save(model.state_dict(), 'resnet18_dual_input.pth')
print("✅ 模型参数已保存为 resnet18_dual_input.pth")