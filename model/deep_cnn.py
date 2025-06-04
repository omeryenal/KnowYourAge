import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepAgeCNN(nn.Module):
    """
    Geliştirilmiş from-scratch CNN.
    - 6 konvolüsyon bloğu (32 → 64 → 128 → 256 → 512 → 512)
    - Her iki blokta bir residual skip bağlantısı
    - BatchNorm + ReLU
    - AdaptiveAvgPool → FC kısmı: 512 → 256 → 128 → 1
    - Dropout (p=0.3)
    - He (Kaiming) ağırlık başlatma
    """

    def __init__(self):
        super(DeepAgeCNN, self).__init__()

        def conv_block(in_ch, out_ch):
            """
            İki konvolüsyon katmanı + BatchNorm + ReLU → çıktı boyutu: [B, out_ch, H/2, W/2]
            """
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # 1. Blok: 3 → 32
        self.block1 = conv_block(3, 32)
        # 2. Blok: 32 → 64
        self.block2 = conv_block(32, 64)
        # 3. Blok: 64 → 128
        self.block3 = conv_block(64, 128)
        # 4. Blok: 128 → 256
        self.block4 = conv_block(128, 256)
        # 5. Blok: 256 → 512
        self.block5 = conv_block(256, 512)
        # 6. Blok: 512 → 512 (ilk conv ile kanal sayısı aynı, residual bağlantı direkt toplanacak)
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adaptive Average Pooling (burada çıktı: [B, 512, 1, 1])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regresyon başlığı (512 → 256 → 128 → 1), dropout p=0.3
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1)
        )

        # Ağırlık başlatma (He initialization)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 3, H, W], varsayalım H=W=128
        x1 = self.block1(x)   # → [B, 32, 64, 64]
        x2 = self.block2(x1)  # → [B, 64, 32, 32]
        x3 = self.block3(x2)  # → [B, 128, 16, 16]
        x4 = self.block4(x3)  # → [B, 256, 8, 8]
        x5 = self.block5(x4)  # → [B, 512, 4, 4]

        # 6. blok ve residual bağlantı
        x6 = self.block6(x5)  # → [B, 512, 2, 2]
        x5_down = F.max_pool2d(x5, kernel_size=2, stride=2)  # → [B, 512, 2, 2]
        x6 = x6 + x5_down    # residual toplanıyor

        # Küresel ortalama havuzlama
        x_pool = self.global_pool(x6)  # → [B, 512, 1, 1]
        x_flat = x_pool.view(x_pool.size(0), -1)  # → [B, 512]

        out = self.fc(x_flat)  # → [B, 1]
        return out.view(-1)    # → [B]
