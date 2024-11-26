import torch
from timm.models import create_model

class ViT(torch.nn.Module):
    def __init__(self, num_classes=10, local_weights_path="./vit_b_16-c867db91.pth"):
        super(ViT, self).__init__()
        
        # 创建 ViT 模型结构（不加载预训练权重）
        self.vit = create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        
        # 加载本地权重文件
        if local_weights_path:
            print(f"Loading weights from {local_weights_path}")
            state_dict = torch.load(local_weights_path, map_location="cpu")
            
            # 如果参数匹配，则加载权重
            self.vit.load_state_dict(state_dict, strict=False)
        else:
            print("No local weights provided, using randomly initialized weights.")

    def forward(self, x):
        return self.vit(x)
