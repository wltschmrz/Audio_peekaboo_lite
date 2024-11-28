import wave
import numpy as np

# 기존 WAV 파일을 읽어서 데이터를 두 번 반복해 저장하는 함수
def duplicate_wav(input_wav, output_wav):
    # WAV 파일 열기
    with wave.open(input_wav, 'rb') as wav_in:
        # WAV 파일의 기본 정보 가져오기
        params = wav_in.getparams()  # (nchannels, sampwidth, framerate, nframes, comptype, compname)
        nchannels, sampwidth, framerate, nframes = params[:4]
        
        # 오디오 데이터 읽기
        audio_data = wav_in.readframes(nframes)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)  # 16-bit PCM 기준
        
        # 오디오 데이터를 두 번 반복
        repeated_audio = np.tile(audio_array, 2)
        
        # 새로운 WAV 파일로 저장
        with wave.open(output_wav, 'wb') as wav_out:
            wav_out.setnchannels(nchannels)
            wav_out.setsampwidth(sampwidth)
            wav_out.setframerate(framerate)
            wav_out.writeframes(repeated_audio.tobytes())

    print(f"10초 WAV 파일이 생성되었습니다: {output_wav}")

# 사용 예시
input_wav = "exp748_cat_mixture.wav"  # 기존 5초 WAV 파일 경로
output_wav = "cat_mixture.wav"  # 생성될 10초 WAV 파일 경로
duplicate_wav(input_wav, output_wav)
