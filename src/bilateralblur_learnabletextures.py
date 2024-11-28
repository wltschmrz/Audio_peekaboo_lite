import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from rp import as_numpy_array

#==========================================================================================================================

# -> (B,2,H,W) [0, 1)
def get_uv_grid(height: int, width: int, batch_size: int = 1) -> torch.Tensor:
    """
    (batch_size, 2, height, width) 크기의 UV grid torch cpu 텐서를 생성합니다.
    UV 좌표는 [0, 1) 사이의 값을 가지며, 1은 포함하지 않아 좌표가 텍스처를 360도로 감싸지 않도록 합니다.
    """

    # 0에서 1까지의 y와 x 좌표를 생성, 끝점은 포함하지 않음
    y_coords = np.linspace(0, 1, height, endpoint=False)  # Shape: (H,)
    x_coords = np.linspace(0, 1, width, endpoint=False)   # Shape: (W,)

    # UV grid 생성 후 예상되는 출력 형태로 reshape
    uv_grid = np.stack(np.meshgrid(y_coords, x_coords), -1)  # Shape: (W, H, 2)    
    uv_grid = torch.tensor(uv_grid).permute(2, 1, 0).unsqueeze(0).float().contiguous()  # Shape: (1, 2, H, W)
    uv_grid = uv_grid.repeat(batch_size, 1, 1, 1)  # batch_size만큼 반복

    return uv_grid  # (1,2,H,W) [0, 1)

######## HELPER FUNCTIONS ########

class GaussianFourierFeatureTransform(nn.Module):
    """
    다음 논문을 참고하여 Gaussian Fourier feature mapping을 구현했음:
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains": https://arxiv.org/abs/2006.10739

    입력 텐서 크기를
    [batches, num_channels, width, height] -> [batches, num_features*2, width, height]
    크기로 변환.
    """
    def __init__(self, num_channels, num_features=256, scale=10):
        """
        Gaussian 분포를 이용해 랜덤한 Fourier components를 생성.
        'scale'이 높을수록 더 높은 frequency features를 생성하여,
        간단한 MLP로도 detailed 이미지를 학습할 수 있지만,
        너무 높을 경우 high-frequency noise만 학습될 위험이 있음.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features

        # 주파수는 Gaussian 분포에서 생성되며, scale로 조정됨 (-> 학습되지 않는 파라미터)
        self.freqs = nn.Parameter(torch.randn(num_channels, num_features) * scale, requires_grad=False)  # (C,F)
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.shape

        # Reshape input for matrix multiplication with freqs: [B, C, H, W] -> [(B*H*W), C]
        x = x.permute(0, 2, 3, 1).reshape(-1, num_channels)

        # Multiply with freqs: [(B*H*W), C] x [C, F] -> [(B*H*W), F]
        x = x @ self.freqs

        # Reshape back to [B, H, W, F] and permute to [B, F, H, W]
        x = x.view(batch_size, height, width, self.num_features).permute(0, 3, 1, 2)

        # Apply sin and cos transformations
        x = 2 * torch.pi * x
        output = torch.cat([torch.sin(x), torch.cos(x)], dim=1)

        return output

######## LEARNABLE IMAGES ########

class LearnableImage(nn.Module):
    def __init__(self, height: int, width: int, num_channels: int):
        super().__init__()
        self.height = height
        self.width = width
        self.num_channels = num_channels
    
    # -> (HWC)np
    def as_numpy_image(self):
        # 이미지를 Numpy 배열 형식으로 변환 반환
        image=self()
        image=as_numpy_array(image)  # (CHW)tensr -> (CHW)np
        image=image.transpose(1,2,0)  # (HWC)np
        return image  # (HWC)np

# -> nn.Param: (C,H,W) [0, 1]
class LearnableImageRaster(LearnableImage):
    def __init__(self, height: int, width: int, num_channels: int):
        super().__init__(height, width, num_channels)
        self.image = nn.Parameter(torch.randn(num_channels, height, width))
        
    def forward(self):
        output = self.image.clone()
        return torch.sigmoid(output)  # (C,H,W)

# -> nn.Conv2d(1conv): (C,H,W) [0, 1]
class LearnableImageFourier(LearnableImage):
    '''
    Fourier features와 MLP를 통해 학습 가능한 이미지를 생성.
    출력 이미지는 각 픽셀 값이 [0, 1] 범위 내의 값.
    '''
    def __init__(self, 
                 height: int, 
                 weight: int, 
                 num_labels: int,
                 hidden_dim: int = 256, 
                 num_features: int = 128, 
                 scale: int = 10):
        
        super().__init__(height, weight, num_labels)

        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.scale = scale
        
        # 학습 중에 변경되지 않는 고정된 UV grid와 Fourier features 추출기
        self.uv_grid = nn.Parameter(get_uv_grid(height, weight, batch_size=1), requires_grad=False)
        self.feature_extractor = GaussianFourierFeatureTransform(2, num_features, scale)
        self.features = nn.Parameter(self.feature_extractor(self.uv_grid), requires_grad=False)

        # MLP 구조를 갖는 1x1 Conv2d로 구성된 모델
        H, C, M = hidden_dim, num_labels, 2 * num_features
        self.model = nn.Sequential(
            nn.Conv2d(M, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, H, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(H),
            nn.Conv2d(H, C, kernel_size=1),
            nn.Sigmoid(),
        )

    def get_features(self, condition=None):
        '''
        Fourier features를 반환하며, 조건(condition)이 주어지면 일부 features를 대체하여 반환합니다.
        !!!:TODO: Don't keep this! Condition should be CONCATENATED! Not replacing features...this is just for testing...
        '''
        features = self.features

        if condition is not None:
            # 첫 n개의 features를 condition으로 대체 (n = len(condition))
            features = features.clone()
            features = rearrange(features, 'B C H W -> B H W C')
            features[..., :len(condition)] = condition
            features = rearrange(features, 'B H W C -> B C H W')

        return features

    def forward(self, condition=None):
        features = self.get_features(condition)
        output = self.model(features).squeeze(0)
        return output  # [num_labels, height, weight]
