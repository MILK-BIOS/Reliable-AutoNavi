import os
import cv2
import socket
import pickle
import asyncio
from langchain_ollama import OllamaLLM
from langchain_core.runnables import Runnable
from ultralytics import YOLO
from .utils import State

class Guardian(Runnable):
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:70b", base_url="http://localhost:11434")
        self.detector = YOLO("/private/workspace/fhs/AN/runs/detect/Traffic_Light_train14/weights/best.pt")
        self.buffer = asyncio.Queue()

    async def load_img(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('10.81.24.156', 2333))
        data = b""
        while True:
            packet = client_socket.recv(4 * 1024)
            if not packet:
                break
            data += packet
            # 解密处理
            decrypted_data = cipher.decrypt(data[8:])
            frame = pickle.loads(decrypted_data)
            self.buffer.append(frame)  # 将解码后的图片加入 buffer
        client_socket.close()
    
    async def process_images(self):
        while True:
            if self.buffer:
                frame = self.buffer.pop(0)  # 从 buffer 中取出一张图片
                # 在这里可以对图片进行处理，例如使用 YOLO 检测
                results = self.detector(frame)
                print("Processed frame:", results)
            await asyncio.sleep(0.1)  # 避免占用过多 CPU 资源

    def invoke(self, state: State, config=None, **kwargs):
        # 异步运行图片加载和处理
        self.load_img(),
        self.process_images()
        return {"status": "Processing completed"}
    
    
    async def ainvoke(self, state: State, config=None, **kwargs):
        # 异步运行图片加载和处理
        await asyncio.gather(
            self.load_img(),
            self.process_images()
        )
        return {"status": "Processing completed"}
    
if __name__  == "__main__":
    guardian = Guardian()
    asyncio.run(guardian.ainvoke())