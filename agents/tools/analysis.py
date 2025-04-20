import cv2
import google.generativeai as genai
import argparse
import time
import base64
from PIL import Image
import io
import requests
from requests.exceptions import RequestException
import os

os.environ['http_proxy'] = 'http://127.0.0.1:7774'
os.environ['https_proxy'] = 'http://127.0.0.1:7774'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7774'

# Gemini API配置
GOOGLE_API_KEY = "AIzaSyDHxlfdq9U9o5w4OpqcHivagdTpv4ZijbU" # 替换为您的API密钥
genai.configure(
    api_key=GOOGLE_API_KEY,
    transport='rest'
    # timeout=30000  # 设置30秒超时
)
model = genai.GenerativeModel("gemini-1.5-flash")

# 设置保存路径
# SAVE_DIR = "G:\\code\\runs\detect\predict"
# os.makedirs(SAVE_DIR, exist_ok=True)

# def check_network_connection():
#     """检查网络连接"""
#     try:
#         response = requests.get("https://generativelanguage.googleapis.com", timeout=5)
#         return True
#     except RequestException as e:
#         print(f"网络连接检查失败: {e}")
#         return False

def process_image_with_gemini(image_path, max_retries=3):
    """使用Gemini处理图片（带重试机制）"""
    # # 检查网络连接
    # if not check_network_connection():
    #     print("网络连接异常，请检查网络设置")
    #     return

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 将OpenCV图片转换为PIL图片
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # 构建提示词
    prompt = """
    请进行场景分析，要描写的如诗如画

    请用中文详尽的描写。
    """
    
    for attempt in range(max_retries):
        try:
            # 发送请求到Gemini
            response =  model.generate_content([prompt, pil_image])
            print("\nGemini分析结果:")
            print(response.text)
            
            # # 获取原始文件名
            # original_filename = os.path.basename(image_path)
            # # 构建新的保存路径
            # output_path = os.path.join(SAVE_DIR, f"analyzed_{original_filename}")
            
            # # 保存图片
            # cv2.imwrite(output_path, frame)
            # print(f"\n图片已保存至: {output_path}")
            
            # # 显示图片
            # cv2.imshow('分析结果', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"处理失败，已达到最大重试次数: {e}")
                return
            print(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
            print("等待2秒后重试...")
            time.sleep(2)

def process_video_with_gemini(source=0, max_retries=3):
    """使用Gemini处理视频流（带重试机制）"""
    # 检查网络连接
    # if not check_network_connection():
    #     print("网络连接异常，请检查网络设置")
    #     return

    # 打开视频源
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("无法打开视频源")
        return

    # 设置显示窗口
    cv2.namedWindow('Gemini实时分析', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gemini实时分析', 1280, 720)

    # 用于控制发送频率的变量
    last_send_time = 0
    SEND_INTERVAL = 5.0  # 每5秒发送一次结果

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧，正在退出")
            break

        # 获取当前时间
        current_time = time.time()
        
        # 每隔一定时间发送帧到Gemini
        if current_time - last_send_time >= SEND_INTERVAL:
            # 将OpenCV图片转换为PIL图片
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 构建提示词
            prompt = """
            请简要描述当前画面中的目标用于路径规划，包括：
            1. 所有可见的目标及其相对于拍摄点的大致方向和距离
            2. 每个目标的类型（如车辆、行人、障碍物）
            3. 每个目标可能的运动情况
            请用简短的中文回答。
            """
            
            for attempt in range(max_retries):
                try:
                    # 发送请求到Gemini
                    response = model.generate_content([prompt, pil_image])
                    print("\nGemini实时分析:", response.text)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"处理失败，已达到最大重试次数: {e}")
                    else:
                        print(f"尝试 {attempt + 1}/{max_retries} 失败: {e}")
                        print("等待2秒后重试...")
                        time.sleep(2)
            
            last_send_time = current_time

        # 显示当前帧
        cv2.imshow('Gemini实时分析', frame)

        # 按下 'q' 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Gemini多模态目标检测')
    parser.add_argument('--source', type=str, default='0', help='输入源：0表示摄像头，或图片/视频文件路径')
    args = parser.parse_args()

    # 判断输入源类型
    if args.source == '0':
        process_video_with_gemini(0)  # 使用摄像头
    else:
        # 检查文件扩展名
        if args.source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            process_image_with_gemini(args.source)  # 处理图片
        else:
            process_video_with_gemini(args.source)  # 处理视频文件

if __name__ == "__main__":
    main()