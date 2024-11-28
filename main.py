from src.peekaboo import run_peekaboo
import torch
import os

from huggingface_hub import login
login(token="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")  # Hugging Face 토큰으로 로그인

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

"""
Peekaboo Hyperparameters:
    GRAVITY: prompt에 따라 tuning이 제일 필요함. 주로 1e-2, 1e-1/2, 1e-1, 1.5*1e-1에서 잘 됨.
    NUM_ITER: 300이면 대부분 충분
    LEARNING_RATE: neural neural textures 아닐 경우, 값 키워도 됨 (1e-5)
    BATCH_SIZE: 큰 차이 없음. 배치 1 키우면 vram만 잡아먹음
    GUIDANCE_SCALE=100: DreamFusion 논문의 고정 값.
    bilateral_kwargs = (kernel_size=3,tolerance=.08,sigma=5,iterations=40)
    square_image_method: input image를 정사각형화 하는 두 가지 방법. (crop / scale)
    representation: (fourier / raster)
"""

results = run_peekaboo(
    name = 'a cat is gently meowing fourth',
    audio = "./wavs/cat_mixture.wav",
    
    GRAVITY = 1e-3,
    NUM_ITER = 300,
    LEARNING_RATE = 1e-5, 
    BATCH_SIZE = 1,   
    GUIDANCE_SCALE = 25,

    representation = 'fourier',

    min_step = None, 
    max_step = None,
)
