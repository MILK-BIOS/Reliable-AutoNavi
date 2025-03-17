import pyttsx3

class VoiceOutput:
    def __init__(self, rate_value=200, volume_value=1.0):
        # 创建语音引擎
        self.model = pyttsx3.init()
        # 设置语速(默认200)
        self.model.setProperty(name="rate",value=rate_value)
        # 设置音量(默认1.0,最大为1.0)
        self.model.setProperty(name="volume", value=volume_value)

    def set(self, rate_value=200, volume_value=1.0):
        # 设置语速(默认200)
        self.model.setProperty(name="rate",value=rate_value)
        # 设置音量(默认1.0,最大为1.0)
        self.model.setProperty(name="volume", value=volume_value)
        
    def run(self, text):
        # 转换为语音
        self.model.say(text)
        # 播放语音
        self.model.runAndWait()

