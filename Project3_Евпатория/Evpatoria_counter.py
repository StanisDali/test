from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import csv
import datetime
import os

cap = cv2.VideoCapture('https://cam.evpanet.com:18120/hls/1942067/411ecfac277d9f587f8f/playlist.m3u8')

csv_file_path = 'car_count.csv'
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Car Count'])

model = YOLO('../Yolo-Weght/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread('mask.png')

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsU = [815, 157, 879, 172]
limitsD = [201, 484, 323, 537]
limitsRD = [848, 409, 786, 554]


totalCount = []

while True:
    success, img = cap.read()
    while not success:
        # Если кадр пустой, ждем получения следующего кадра
        success, img = cap.read()
        # Дополнительная проверка на переподключение к видео по ссылке
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture('https://cam.evpanet.com:18120/hls/1942067/411ecfac277d9f587f8f/playlist.m3u8')

    img = cv2.resize(img, (1280,720))

    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0,0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes # здесь создание рамок идет (2 варианта), в cornerRect через контрл можно поменять стиль рамки
        for box in boxes:
            #Boundingbox
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)
            w, h = x2-x1, y2-y1
            #Confidence
            conf = math.ceil((box.conf[0]*100))/100 # show how we shure about object
            #class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus'\
                    or currentClass == 'motorbike' or currentClass == "bicycle" and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=1,thickness=1,offset=5)
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt= 5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

                resultsTracker = tracker.update(detections)
                lineU = cv2.line(img, (limitsU[0],limitsU[1]), (limitsU[2], limitsU[3]), (0, 0, 255), 5)
                lineD = cv2.line(img, (limitsD[0], limitsD[1]), (limitsD[2], limitsD[3]), (0, 0, 255), 5)
                lineRD = cv2.line(img, (limitsRD[0], limitsRD[1]), (limitsRD[2], limitsRD[3]), (0, 0, 255), 5)

                for result in resultsTracker:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    #print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,255))
                    #cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                    #                   scale=1, thickness=2, offset=5)

                    cx,cy = x1+w//2, y1+h//2
                    cv2.circle(img, (cx, cy), 5, (255,0, 255), cv2.FILLED)

                    if limitsU[0]<cx<limitsU[2] and limitsU[1]-20<cy<limitsU[1]+20:
                        if totalCount.count(id) == 0:
                            totalCount.append(id)
                            cv2.line(img, (limitsU[0], limitsU[1]), (limitsU[2], limitsU[3]), (0, 255, 0), 5)
                            # Записываем в CSV текущее время и количество автомобилей
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            with open(csv_file_path, 'a', newline='', encoding='cp1251') as csv_file:
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow([timestamp, len(totalCount), currentClass, "Up"])

                    elif limitsD[0]<cx<limitsD[2] and limitsD[1]-20<cy<limitsD[1]+20:
                        if totalCount.count(id) == 0:
                            totalCount.append(id)
                            cv2.line(img, (limitsD[0], limitsD[1]), (limitsD[2], limitsD[3]), (0, 255, 0), 5)
                            # Записываем в CSV текущее время и количество автомобилей
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            with open(csv_file_path, 'a', newline='', encoding='cp1251') as csv_file:
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow([timestamp, len(totalCount), currentClass, "Down"])

                    elif limitsRD[0]+10>cx>limitsRD[2]-10 and limitsRD[1]<cy<limitsRD[3]:
                        if totalCount.count(id) == 0:
                            totalCount.append(id)
                            cv2.line(img, (limitsRD[0], limitsRD[1]), (limitsRD[2], limitsRD[3]), (0, 255, 0), 5)
                            # Записываем в CSV текущее время и количество автомобилей
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            with open(csv_file_path, 'a', newline='') as csv_file:
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow([timestamp, len(totalCount), currentClass, "Right"])

        #cvzone.putTextRect(img, f'Count{len(totalCount)}', (50,50))
        cv2.putText(img, str(len(totalCount)), (255,100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow('Image', img)
 #   cv2.imshow('ImageRegion', imgRegion)
    cv2.waitKey(1)