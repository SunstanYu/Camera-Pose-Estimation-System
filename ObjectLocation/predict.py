import cv2
from ultralytics import YOLO
import PIL.Image as Image
import os

# model = YOLO("models/best.pt")  # 加载预训练的 YOLOv8n 模型
#
# # model = YOLO("yolov8n.pt")  # load an official model
#
# # Predict with the model and initail
# results = model("origin_img/IMG_2213.JPG")  # predict on an image
# # 标签
#
#
# lables = results[0].names


def getboxs(yoloresult, lables):
    # 获取识别框信息
    BOXS = []
    outputs = yoloresult[0].boxes
    class_IDs = outputs.cls.tolist()
    layables = []
    for i in class_IDs:
        layables.append(lables[int(i)])

    confidences = outputs.conf.tolist()
    boxes = outputs.xyxy.tolist()
    if len(boxes):
        BOXS = [boxes, layables, confidences]

    return BOXS


def drawbox(img, box, filte=0.5):
    if type(box) == type([]):
        if len(box) > 0:
            for i in range(len(box[0])):
                x, y, x1, y1 = box[0][i]
                x = int(x)
                y = int(y)
                x1 = int(x1)
                y1 = int(y1)
                name = box[1][i]
                confi = box[2][i]
                if confi >= filte:
                    text = "{}: {:.4f}".format(name, confi)
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_ITALIC, 0.5, [0, 255, 0], 2)
                    cv2.rectangle(img, (x, y), (x1, y1), (255, 255, 0), 2)

# cap=cv2.VideoCapture(0)
# video_viewer(cap)

# 加载并预处理图像
# test_path = "origin_img"
# result = model.predict(test_path)
# print(result)
#     # 检查文件是否是图片文件
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         img = Image.open(os.path.join(test_path, filename))
#         image = transform(img)
#         with torch.no_grad():
#             predictions = model([image])
#         # 解析预测结果
#         boxes = predictions[0]['boxes']
#         scores = predictions[0]['scores']
#         labels = predictions[0]['labels']
#
#         # 打印检测到的物体信息
#         for box, score, label in zip(boxes, scores, labels):
#             print(f"物体类别: {label}, 置信度: {score}, 边界框: {box}\n")
#
