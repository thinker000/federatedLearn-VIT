import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class GetDataSet:
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_set = None
        self.test_set = None

        if self.name == 'cifar10':
            self.cifar10DataSetConstruct(isIID)
        else:
            raise ValueError(f"Unsupported dataset: {self.name}")

    def cifar10DataSetConstruct(self, isIID):
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整到 ViT 所需尺寸
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
        ])

        # 加载 CIFAR-10 数据集
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # 如果是非 IID，按标签排序
        if not isIID:
            train_set.targets, train_set.data = zip(*sorted(zip(train_set.targets, train_set.data), key=lambda x: x[0]))

        self.train_set = train_set
        self.test_set = test_set

    def getTrainLoader(self, batch_size):
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=True)

    def getTestLoader(self, batch_size):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False)
