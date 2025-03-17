import pyaudio
from vosk import Model, KaldiRecognizer
import json

class VoiceInput:
    def __init__(self, format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000):
        # 加载模型
        self.vosk_model = Model('/private/workspace/fhs/AN/model/vosk-model-cn-0.22')
        # 创建麦克风对象
        self.mic = pyaudio.PyAudio()
        self.mic_buffer = self.mic.open(
        # 16位深度音频数据
        format=format,
        # 声道，单声道
        channels=channels,
        # 采样率
        rate=rate,
        # 从麦克风获取数据
        input=input,
        # 每次读取数据块大小
        frames_per_buffer=frames_per_buffer
        )
        # 创建语音识别器
        self.voice_recognizer = KaldiRecognizer(self.vosk_model, 16000)
        
    def run(self):
        print('开始实时识别')

        while True:
            # 从麦克风读取数据
            voice_data = self.mic_buffer.read(4000)
            # 如果读取到数据
            if self.voice_recognizer.AcceptWaveform(voice_data):
                # 实时输出识别结果
                output = json.loads(self.voice_recognizer.Result())['text'].replace(' ', '')
                print(output)
