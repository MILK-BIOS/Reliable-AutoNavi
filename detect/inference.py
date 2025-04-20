import cv2
import os
from ultralytics import YOLO

# 初始化 YOLO 模型

class YOLODetector:
    def __init__(self, model='yolov8l-world.pt'):
        self.model = YOLO("/private/workspace/fhs/AN/agents/yolov8l-world.pt")  # 或选择 yolov8m/l-world.pt

        # 定义自定义类别
        self.model.set_classes(["Person"])

        # 定义输入和输出文件夹
        self.input_folder = "work_dir/uploaded_frames"
        self.output_folder = "work_dir/detect_results"

        # 创建输出文件夹（如果不存在）
        os.makedirs(self.output_folder, exist_ok=True)

    def set_classes(self, classes):
        self.model.set_classes(classes)

    def inference(self, frame, conf=0.5):
        # 遍历输入文件夹中的所有图片
        for filename in os.listdir(self.input_folder):
            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, filename)

            # 检查文件是否为图片
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"跳过非图片文件: {filename}")
                continue

            # 读取图片
            frame = cv2.imread(input_path)
            if frame is None:
                print(f"无法读取图片: {input_path}")
                continue

            # 使用 YOLO 模型进行预测
            results = self.model.predict(source=frame, show=False, conf=conf)  # 设置置信度阈值

            # 绘制检测结果
            annotated_frame = results[0].plot()  # 绘制检测框和标签

            # 保存检测结果到输出文件夹
            cv2.imwrite(output_path, annotated_frame)
            # print(f"检测结果已保存: {output_path}")
            try:
                os.remove(input_path)
                print(f"已删除已处理的图片: {input_path}")
            except Exception as e:
                print(f"删除图片时发生错误: {e}")
