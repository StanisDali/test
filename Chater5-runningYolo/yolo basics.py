from ultralytics import YOLO
import cv2

model = YOLO("../Yolo-Weight/yolov8n.pt")
results = model("Images/3.png", show=True)
cv2.waitKey(0)