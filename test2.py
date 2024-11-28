from diffusers import AudioLDMPipeline

pipeline = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2")
print(pipeline.text_encoder.embed_mode)  # 현재 설정 확인
