from typing import Union, List, Optional
from transformers import logging
from diffusers import PNDMScheduler
from diffusers import AudioLDMPipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from src.audio import wav_to_fbank, TacotronSTFT, read_wav_file
from src.utils import default_audioldm_config, get_duration, get_bit_depth, round_up_duration


# Suppress partial model loading warning
logging.set_verbosity_error()


class AudioLDM(nn.Module):
    def __init__(self, device='cuda', checkpoint_path="cvssp/audioldm-s-full"):
        super().__init__()

        self.device = torch.device(device)
        self.num_train_timesteps = 1000

        # Timestep ~ U(0.02, 0.98) to avoid very high/low noise levels
        self.min_step = int(self.num_train_timesteps * 0.02)  # aka 20
        self.max_step = int(self.num_train_timesteps * 0.98)  # aka 980

        # Unlike the original code, I'll load these from the pipeline. This lets us use DreamBooth models.
        pipe = AudioLDMPipeline.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16, ##
            safety_checker=None,
        ).to(self.device)

        pipe.scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=self.num_train_timesteps) #Error from scheduling_lms_discrete.py
        
        self.pipe         = pipe
        self.vae          = pipe.vae.to(self.device)           # AutoencoderKL
        self.tokenizer    = pipe.tokenizer                     # Union[RobertaTokenizer, RobertaTokenizerFast]
        self.text_encoder = pipe.text_encoder.to(self.device)  # ClapTextModelWithProjection
        self.unet         = pipe.unet.to(self.device)          # UNet2DConditionModel
        self.vocoder      = pipe.vocoder.to(self.device)       # SpeechT5HifiGan
        self.scheduler    = pipe.scheduler                     # PNDMScheduler

        self.checkpoint_path = checkpoint_path
        self.config = default_audioldm_config(model_name='audioldm-s-full')
        self.uncond_text = ''
        '''
        - 빈 문자열 (''): 
            텍스트 조건이 전혀 필요하지 않은 상황에서는 빈 문자열을 사용하여 임베딩이 최소한의 영향을 받도록 할 수 있습니다.
        - 일반적인 텍스트 (' ', 'None' 등): 
            오디오 도메인에서 특정 조건이 없음을 의미하는 텍스트를 설정할 수 있습니다.
            예를 들어, 음성 생성 모델의 경우,
            'silence', 'background noise', 'ambient'와 같은 텍스트를 조건으로 추가해볼 수 있습니다.
        '''
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        print(f'[INFO] Loaded AudioLDM model from {checkpoint_path}')


    def get_text_embeddings(self, prompts: Union[str, List[str]], num_waveforms_per_prompt=1) -> torch.Tensor:

        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
            prompts = [prompts]
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            raise ValueError
            
        text_inputs = self.tokenizer(
            prompts,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt'
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
            prompt_embeds = prompt_embeds.text_embeds
            prompt_embeds = F.normalize(prompt_embeds, dim=-1)
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

            bs_embed, seq_len = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)

        uncond_tokens: List[str]
        uncond_tokens = [""] * batch_size
        max_length = prompt_embeds.shape[1]
        
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(self.device)
        attention_mask = uncond_input.attention_mask.to(self.device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds.text_embeds
            negative_prompt_embeds = F.normalize(negative_prompt_embeds, dim=-1)

            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len)

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds  # dim?

    
    def train_step(self,
                   text_embeddings: torch.Tensor,
                   pred_audio: torch.Tensor,  # B3HW
                   guidance_scale: float = 100,
                   t: Optional[int] = None):
        '이 메서드는 dream-loss gradients을 생성하는 역할.'

        input_mel = torch.clamp(pred_audio[:, :1] * 9 - 9, -9, 0)
        input_mel = input_mel.half() 

        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # VAE를 사용하여 이미지를 latents로 인코딩 / grad 필요.
        latents = self.encode_mels(input_mel)

        # unet으로 noise residual 예측 / NO grad.
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t.cpu())
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=None, class_labels=text_embeddings).sample

        # guidance 수행 (high scale from paper)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2   
        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # grad에서 item을 생략하고 자동 미분 불가능하므로, 수동 backward 수행.
        latents.backward(gradient=grad, retain_graph=True)
        return 0  # dummy loss value

    def encode_mels(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        
        posterior = self.vae.encode(mel_spectrogram)  # BCHW -> (B,8,8,256)
        latent_distribution = posterior.latent_dist

        if isinstance(latent_distribution, DiagonalGaussianDistribution):
            latents = latent_distribution.sample()
        elif isinstance(latent_distribution, torch.Tensor):
            latents = latent_distribution
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(latent_distribution)}' not yet implemented"
            )
        latents = latents * self.vae.config.scaling_factor

        if(torch.max(torch.abs(latents)) > 1e2):
            latents = torch.clip(latents, min=-10, max=10)
        return latents  # (B,8,8,256)

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def waveform_to_mel_spectrogram(self, waveform_path, duration=10, batchsize=1):
        
        assert waveform_path is not None, "You need to provide the original audio file path"
        
        audio_file_duration = get_duration(waveform_path)

        assert get_bit_depth(waveform_path) == 16, "The bit depth of the original audio file %s must be 16" % waveform_path
        
        if(duration > audio_file_duration):
            print("Warning: Duration you specified %s-seconds must equal or smaller than the audio file duration %ss" % (duration, audio_file_duration))
            duration = round_up_duration(audio_file_duration)
            print("Set new duration as %s-seconds" % duration)

        fn_STFT = TacotronSTFT(
            self.config["preprocessing"]["stft"]["filter_length"],
            self.config["preprocessing"]["stft"]["hop_length"],
            self.config["preprocessing"]["stft"]["win_length"],
            self.config["preprocessing"]["mel"]["n_mel_channels"],
            self.config["preprocessing"]["audio"]["sampling_rate"],
            self.config["preprocessing"]["mel"]["mel_fmin"],
            self.config["preprocessing"]["mel"]["mel_fmax"],
        )

        mel, _, _ = wav_to_fbank(waveform_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT)  # HW
        mel = mel.unsqueeze(0).unsqueeze(0)  # 11HW
        mel = repeat(mel, "1 ... -> b ...", b=batchsize)  # B1HW

        return mel

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        waveform = waveform.cpu().float()
        return waveform
    
