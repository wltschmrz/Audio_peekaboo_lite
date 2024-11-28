from diffusers import AudioLDMPipeline
import torch
import soundfile as sf
import rp
from src.peekaboo import PeekabooResults
from src.peekaboo import save_peekaboo_results

from huggingface_hub import login
login(token="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")  # Hugging Face 토큰으로 로그인

checkpoint_path = 'cvssp/audioldm-s-full-v2'
device = 'cuda'


pipe = AudioLDMPipeline.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16, ##
            safety_checker=None,
        ).to(device)

wav, mel, aud = pipe('a cat is gently meowing twice')  # aud가 wav파일인데 (1,81920), mel은 (1,1,512,64)



aud = aud.squeeze(0)
mel = mel.squeeze(0).permute(0,2,1)
mel = torch.clamp((mel + 9) / 9, 0, 1)  # (3,H,W) [0,1] normalizing

sf.write(f'./output_sound.wav', aud, 16000, subtype='PCM_16')
results = PeekabooResults(
        #The main output
        alphas=rp.as_numpy_array(mel),)
output_folder = rp.make_folder(f'peekaboo_results/test')
output_folder += f'/{len(rp.get_subfolders(output_folder)):03}'
save_peekaboo_results(results, output_folder)

'''
여러번 시도하다가 확인된 사항
-> 단어 단위 prompt는 성능이 굉장히 안좋다. 완벽한 문장 형태로 prompt를 주어야 generation을 월등히 잘한다.

'''