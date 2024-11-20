from typing import Union, List, Optional
import numpy as np
import rp
import torch
import torch.nn as nn
from easydict import EasyDict
from einops import repeat

import src.audioldm as aldm
from src.bilateralblur_learnabletextures import (BilateralProxyBlur,
                                    LearnableImageFourier,
                                    LearnableImageFourierBilateral,
                                    LearnableImageRaster,
                                    LearnableImageRasterBilateral)

#Importing this module loads a stable diffusion model. Hope you have a GPU
aldm = aldm.AudioLDM(device='cuda',
                     checkpoint_path='cvssp/audioldm-s-full-v2',)
device = aldm.device


#==========================================================================================================================

def make_learnable_image(height, weight, num_channels, foreground=None, bilateral_kwargs=None, representation='fourier'):
    "이미지의 파라미터화 방식을 결정하여 학습 가능한 이미지를 생성."
    bilateral_kwargs = bilateral_kwargs or {}
    bilateral_blur = BilateralProxyBlur(foreground, **bilateral_kwargs)
    if representation == 'fourier bilateral':
        return LearnableImageFourierBilateral(bilateral_blur, num_channels)
    elif representation == 'raster bilateral':
        return LearnableImageRasterBilateral(bilateral_blur, num_channels)
    elif representation == 'fourier':
        return LearnableImageFourier(height, weight, num_channels)
    elif representation == 'raster':
        return LearnableImageRaster(height, weight, num_channels)
    else:
        raise ValueError(f'Invalid method: {representation}')

def blend_torch_images(foreground, background, alpha):
    '주어진 foreground와 background 이미지를 alpha 값에 따라 블렌딩합니다.'
    C, H, W = foreground.shape
    assert foreground.shape == background.shape, 'foreground, background shape 다름'
    assert alpha.shape == (H, W), 'alpha가 (H, W) 크기의 행렬이 아님.'
    return foreground * alpha + background * (1 - alpha)

def make_image_square(image: np.ndarray, method='crop') -> np.ndarray:
    """
    주어진 이미지를 512x512(x3) 크기의 정사각형으로 변환. 
    method는 'crop' 또는 'scale' 중 하나를 사용할 수 있다."""
    
    try:
        image = rp.as_rgb_image(image)  # 이미지가 3채널 RGB인지 확인 및 변환
        height, width = rp.get_image_dimensions(image)
        min_dim = min(height, width)
    except:
        height, width = rp.get_image_dimensions(image)
        min_dim = min(height, width)

    if method == 'crop':
        # 중앙에서 자르고 스케일링하여 정사각형 이미지 생성
        return make_image_square(rp.crop_image(image, min_dim, min_dim, origin='center'), 'scale')
    
    # 'scale' 메서드일 경우 이미지 크기를 512x512로 조정
    return rp.resize_image(image, (512, 512))

#==========================================================================================================================

class PeekabooResults(EasyDict):
    pass

class BaseLabel:
    def __init__(self, name:str, embedding:torch.Tensor):
        self.name=name
        self.embedding=embedding

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"
        
class SimpleLabel(BaseLabel):
    def __init__(self, name:str):
        super().__init__(name, aldm.get_text_embeddings(name).to(device))

#==========================================================================================================================

def save_peekaboo_results(results, new_folder_path):
    'PeekabooResults를 지정된 폴더에 저장.'
    import json
    assert not rp.folder_exists(new_folder_path), f'Please use a different name, not {new_folder_path}'
    rp.make_folder(new_folder_path)

    with rp.SetCurrentDirectoryTemporarily(new_folder_path):
        print(f"\nSaving PeekabooResults to {new_folder_path}")
        params = {}

        for key, value in results.items():
            if rp.is_image(value):
                rp.save_image(value, f'{key}.png')  # 단일 이미지 저장
            elif isinstance(value, np.ndarray) and rp.is_image(value[0]):
                rp.make_directory(key)  # 이미지 폴더 저장
                with rp.SetCurrentDirectoryTemporarily(key):
                    [rp.save_image(img, f'{i}.png') for i, img in enumerate(value)]
            elif isinstance(value, np.ndarray):
                np.save(f'{key}.npy', value)  # 일반 Numpy 배열 저장
            else:
                try:
                    json.dumps({key: value})  # JSON으로 변환 가능한 값 저장
                    params[key] = value
                except Exception:
                    params[key] = str(value)  # 변환 불가한 값은 문자열로 저장

        rp.save_json(params, 'params.json', pretty=True)
        print(f"Done saving PeekabooResults to {new_folder_path}!")

#==========================================================================================================================

class PeekabooSegmenter(nn.Module):
    '이미지 분할을 위한 PeekabooSegmenter 클래스.'
    
    def __init__(self, 
                 audio: np.ndarray, 
                 labels: List['BaseLabel'], 
                 mel_channel: int = 64,  # C
                 target_length: int = 1024,  # T
                 channel: int = 1,
                 name: str = 'Untitled', 
                 bilateral_kwargs: dict = None, 
                 representation: str = 'fourier bilateral', 
                 min_step=None, 
                 max_step=None):
        super().__init__()     

        self.height = mel_channel
        self.width = target_length
        self.channel = channel
        self.labels = labels
        self.name = name
        self.representation = representation
        self.min_step = min_step
        self.max_step = max_step
        

        mel = audio  # B1HW
        hwc_tensor = repeat(mel[0, 0], "h w -> c h w", c=3)  # (3,H,W)
        normalized_tensor = torch.clamp((hwc_tensor + 9) / 9, 0, 1)  # (3,H,W) [0,1] normalizing
        
            # # 이미지 전처리
            # image = rp.cv_resize_image(image, (self.height, self.width))  # np (256,256,3)

            # # 채널에 맞게 이미지 변환
            # if self.channel == 3:
            #     image = rp.as_rgb_image(image)  # 3채널로 변환 (RGB)

            # image = rp.as_float_image(image)  # 값의 범위를 0과 1 사이로 변환
            # self.image = image  # np (256,256,3) & norm

        img = repeat(mel[0, 0], "h w -> h w c", c=3)  # (H,W,3)
        img = torch.clamp((img + 9) / 9 * 255, 0, 255)  # [0,255] normalizing
        # croped = make_image_square(normalized_tensor, square_image_method)
        img = rp.as_numpy_array(img.to(torch.uint8))
        mel_image = rp.as_rgb_image(rp.as_float_image(img))  # -> np (512,512,3)
        self.image = mel_image

        # 이미지를 Torch 텐서로 변환 (CHW 형식)
        self.foreground = rp.as_torch_image(normalized_tensor).to(device)  # torch (3,64,1024) & norm
        
        # 배경은 단색으로 설정
        self.background = torch.zeros_like(self.foreground)
        
        # 학습 가능한 알파 값 생성
        bilateral_kwargs = bilateral_kwargs or {}
        self.alphas = make_learnable_image(self.height,
                                           self.width,
                                           num_channels=len(labels), 
                                           foreground=self.foreground, 
                                           representation=self.representation, 
                                           bilateral_kwargs=bilateral_kwargs)
            
    @property
    def num_labels(self):
        return len(self.labels)
            
    def set_background_color(self, color):
        r,g,b = color
        self.background[0] = r
        self.background[1] = g
        self.background[2] = b

    def default_background(self):
        self.background[0] = 0
        self.background[1] = 0
        self.background[2] = 0
        
    def forward(self, alphas=None, return_alphas=False):        
        # alpha 값이 없으면, 학습된 alpha 생성
        alphas = alphas if alphas is not None else self.alphas()

        # alpha 값을 이용하여 각 라벨에 대한 이미지를 생성
        output_images = [blend_torch_images(self.foreground, self.background, alpha) for alpha in alphas]
        output_images = torch.stack(output_images)

        return (output_images, alphas) if return_alphas else output_images

def display(self):

    # 단일 채널인지 다중 채널인지 확인
    if self.channel == 1:
        colors = [0]  # 단일 채널은 색상이 아니라 단색 값
        self.default_background()
        composites = [rp.as_numpy_images(self(self.alphas()))]  # 기본 배경 사용
    else:
        colors = [rp.random_rgb_float_color() for _ in range(3)]
        composites = [rp.as_numpy_images(self(self.alphas())) for color in colors for _ in [self.set_background_color(color)]]
    
    # alpha 채널을 numpy 배열로 변환
    alphas = rp.as_numpy_array(self.alphas())

    # 레이블 이름 및 상태 정보 설정
    label_names = [label.name for label in self.labels]
    stats_lines = [self.name, '', f'H,W = {self.height}x{self.width}']

    # 전역 변수에서 특정 상태 정보를 추가
    for stat_format, var_name in [('Gravity: %.2e', 'GRAVITY'),
                                  ('Batch Size: %i', 'BATCH_SIZE'),
                                  ('Iter: %i', 'iter_num'),
                                  ('Image Name: %s', 'image_filename'),
                                  ('Learning Rate: %.2e', 'LEARNING_RATE'),
                                  ('Guidance: %i%%', 'GUIDANCE_SCALE')]:
        if var_name in globals():
            stats_lines.append(stat_format % globals()[var_name])

    # 기본 배경 또는 랜덤 배경과 함께 출력 이미지 생성
    if self.channel == 1:
        # 단일 채널 이미지 처리
        output_image = rp.labeled_image(
            rp.tiled_images(
                rp.labeled_images(
                    [self.image, alphas[0], composites[0][0]],
                    ["Input Image", "Alpha Map", "Default Background"]
                ),
                length=3  # 단일 채널은 3개의 이미지
            ),
            label_names[0]
        )
    else:
        # 다중 채널 이미지 처리
        output_image = rp.labeled_image(
            rp.tiled_images(
                rp.labeled_images(
                    [self.image,
                     alphas[0],
                     composites[0][0],
                     composites[1][0],
                     composites[2][0]],
                    ["Input Image",
                     "Alpha Map",
                     "Background #1",
                     "Background #2",
                     "Background #3"],
                ),
                length=2 + len(composites),
            ),
            label_names[0]
        )
    # 이미지 출력
    rp.display_image(output_image)
    return output_image
PeekabooSegmenter.display = display

def run_peekaboo(name: str,
                 audio: Union[str, np.ndarray],
                 label: Optional['BaseLabel'] = None,

                 GRAVITY=1e-1/2,
                 NUM_ITER=300,
                 LEARNING_RATE=1e-5, 
                 BATCH_SIZE=1,   
                 GUIDANCE_SCALE=100,
                 bilateral_kwargs=dict(
                     kernel_size=3,
                     tolerance=0.08,
                     sigma=5,
                     iterations=40
                     ),
                 square_image_method='crop', 
                 representation='fourier bilateral',
                 min_step=None, 
                 max_step=None) -> PeekabooResults:
    """
    Peekaboo Hyperparameters:
        GRAVITY: prompt에 따라 tuning이 제일 필요함. 주로 1e-2, 1e-1/2, 1e-1, 1.5*1e-1에서 잘 됨.
        NUM_ITER: 300이면 대부분 충분
        LEARNING_RATE: neural neural textures 아닐 경우, 값 키워도 됨
        BATCH_SIZE: 큰 차이 없음. 배치 1 키우면 vram만 잡아먹음
        GUIDANCE_SCALE=100: DreamFusion 논문의 고정 값.
        bilateral_kwargs = (kernel_size=3,tolerance=.08,sigma=5,iterations=40)
        square_image_method: input image를 정사각형화 하는 두 가지 방법. (crop / scale)
        representation: (fourier bilateral / raster bilateral / fourier / raster)
    """
    
    # 레이블이 없을 경우 기본 레이블 생성
    label = label or SimpleLabel(name)  # label이 갖고 있는 embedding dim은 (2,77,768) -> (2,)

    # # 이미지 로드 및 전처리
    # image_path = image if isinstance(image, str) else '<No image path given>'
    # image = rp.load_image(image_path) if isinstance(image, str) else image  # np (500,500,3)
    # image = rp.as_rgb_image(rp.as_float_image(make_image_square(image, square_image_method)))  # np (512,512,3)

    # 오디오 로드 및 전처리
    audio_path = audio if isinstance(audio, str) else '<No image path given>'
    mel = aldm.waveform_to_mel_spectrogram(audio_path, duration=10, batchsize=BATCH_SIZE)  # B1HW

    hwc_tensor = repeat(mel[0, 0], "h w -> h w c", c=3)  # (H,W,3)
    normalized_tensor = torch.clamp((hwc_tensor + 9) / 9 * 255, 0, 255)  # [0,255] normalizing
    # croped = make_image_square(normalized_tensor, square_image_method)
    normalized_tensor_uint8 = rp.as_numpy_array(normalized_tensor.to(torch.uint8))

    mel_image = rp.as_rgb_image(rp.as_float_image(normalized_tensor_uint8))  # -> np (512,512,3)
    
    # PeekabooSegmenter 생성
    pkboo = PeekabooSegmenter(
        mel,
        labels=[label],
        name=name,
        bilateral_kwargs=bilateral_kwargs,
        representation=representation,
        min_step=min_step,
        max_step=max_step
        ).to(device)

    pkboo.display()

    # 옵티마이저 설정
    params = list(pkboo.parameters())
    optim = torch.optim.SGD(params, lr=LEARNING_RATE)

    # 학습 반복 설정
    global iter_num
    iter_num = 0
    timelapse_frames=[]
    preview_interval = max(1, NUM_ITER // 10)  # 10번의 미리보기를 표시

    try:
        display_eta = rp.eta(NUM_ITER)
        for _ in range(NUM_ITER):
            display_eta(_)
            iter_num += 1

            alphas = pkboo.alphas()
            for __ in range(BATCH_SIZE):
                pkboo.default_background()
                composites = pkboo()
                for label, composite in zip(pkboo.labels, composites):
                    aldm.train_step(label.embedding, composite[None], guidance_scale=GUIDANCE_SCALE)

            ((alphas.sum()) * GRAVITY).backward()
            optim.step()
            optim.zero_grad()

            with torch.no_grad():
                if not _ % preview_interval: 
                    timelapse_frames.append(pkboo.display())

    except KeyboardInterrupt:
        print("Interrupted early, returning current results...")
                

    results = PeekabooResults(
        #The main output
        alphas=rp.as_numpy_array(alphas),
        
        #Keep track of hyperparameters used
        GRAVITY=GRAVITY, BATCH_SIZE=BATCH_SIZE, NUM_ITER=NUM_ITER, GUIDANCE_SCALE=GUIDANCE_SCALE,
        bilateral_kwargs=bilateral_kwargs, representation=representation, label=label,
        audio_image=mel_image, audio_path=audio_path, 
        
        #Record some extra info
        preview_image=pkboo.display(), timelapse_frames=rp.as_numpy_array(timelapse_frames),
        height=pkboo.height, width=pkboo.width, p_name=pkboo.name, min_step=pkboo.min_step, max_step=pkboo.max_step,) 
    
    # 결과 폴더 생성 및 저장
    output_folder = rp.make_folder(f'peekaboo_results/{name}')
    output_folder += f'/{len(rp.get_subfolders(output_folder)):03}'
    save_peekaboo_results(results, output_folder)
  