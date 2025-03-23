import cryptography
import socket
import pickle
from cryptography.fernet import Fernet
import asyncio
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import base64
import struct
import logging

logging.basicConfig(level=logging.INFO)

buffer = asyncio.Queue()
with open("secret.key", "rb") as key_file:
    key = key_file.read()
cipher = Fernet(key)
# detector = YOLO("/private/workspace/fhs/AN/runs/detect/Traffic_Light_train14/weights/best.pt")
detector = YOLO("yolov10l")
executor = ThreadPoolExecutor(max_workers=4)

async def load_img():
    reader, writer = None, None
    try:
        logging.info("Connecting to server...")
        reader, writer = await asyncio.open_connection('10.81.24.156', 2333)
        data = b""
        while True:
            # 异步读取数据
            packet = await reader.read(4 * 1024)
            if not packet:
                logging.warning("Connection closed by the server.")
                break
            data += packet

            # 解析数据
            while len(data) >= 8:  # 确保有足够的数据解析长度
                msg_len = struct.unpack("!Q", data[:8])[0]
                if len(data) < 8 + msg_len:
                    break
                encrypted_data = data[8:8 + msg_len]
                data = data[8 + msg_len:]
                try:
                    decrypted_data = cipher.decrypt(encrypted_data)
                    frame = pickle.loads(decrypted_data)
                    await buffer.put(frame)
                    logging.info(f"Frame added to buffer. Buffer size: {buffer.qsize()}")
                except (cryptography.fernet.InvalidToken, pickle.UnpicklingError) as e:
                    logging.error(f"Decryption or deserialization error: {e}")
    except Exception as e:
        logging.error(f"Error in load_img: {e}")
    finally:
        if writer:
            writer.close()
            await writer.wait_closed()

async def process_images():
    try:
        logging.info("Starting process_images...")
        loop = asyncio.get_event_loop()
        while True:
            logging.info("Waiting for frame from buffer...")
            frame = await buffer.get()
            logging.info("Frame received from buffer.")
            results = await loop.run_in_executor(executor, detector, frame)
            logging.info("Processed frame: %s", results)
    except Exception as e:
        logging.error(f"Error in process_images: {e}")

async def main():
    await asyncio.gather(
        load_img(),
        process_images(),
        return_exceptions=True
    )

if __name__ == "__main__":
    asyncio.run(main())