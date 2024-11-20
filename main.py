from src.peekaboo import run_peekaboo
import torch
import os

from huggingface_hub import login
login(token="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")  # Hugging Face 토큰으로 로그인

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

results = run_peekaboo(
    'water drops',
    "extended_water_drops.wav"
)
