import cv2
import numpy as np
from ultralytics import YOLO
import math
import json

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

#Porcentaje de reducción del video
resize_factor = 0.5

with open("coordinates.json", 'r') as file:
    coordinates = json.load(file)

list_of_coordinates = []

for coord in coordinates:
    x = int(coord['x'])
    y = int(coord['y'])
    list_of_coordinates.append((x, y))

c_x1 = list_of_coordinates[0][0]
c_y1 = list_of_coordinates[0][1]
c_x2 = list_of_coordinates[1][0]
c_y2 = list_of_coordinates[1][1]

# Coordenadas originales del video completo
custom_x1, custom_y1 = int(c_x1 * resize_factor), int(c_y1 * resize_factor)
custom_x2, custom_y2 = int(c_x2 * resize_factor), int(c_y2 * resize_factor)


# Cargar el modelo
model = YOLO("yolov8n.pt")

video_path = "/Users/caballero/Desktop/Servicio Social/carCounter/video.mp4"
cap = cv2.VideoCapture(video_path)

count=0

previous_centers = []
counted_centers = set()

distance_threshold = 50  # px

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    height, width = frame.shape[:2]

    results = model.track(frame, persist=True)
    annotated_frame = frame.copy()

    # Dibujar la línea personalizada
    cv2.line(annotated_frame, (custom_x1, custom_y1), (custom_x2, custom_y2), (0, 0, 255), 2)

    current_centers = []

    boxes = results[0].boxes
    if boxes is not None:
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            if class_id == 2:  # Carro
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                score = boxes.conf[i].item()
                label = f"Car {score:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                current_centers.append((cx, cy))

                cv2.circle(annotated_frame, (cx, cy), 3, (255, 0, 0), -1)


    for c in current_centers:
        min_dist = float('inf')
        closest_prev = None
        for pc in previous_centers:
            dist = euclidean_distance(c, pc)
            if dist < min_dist:
                min_dist = dist
                closest_prev = pc

        if min_dist < distance_threshold and closest_prev is not None:
            #cambiar a 0 si la linea es vertical
            prev_x = closest_prev[0]
            prev_y = closest_prev[1]
            curr_x = c[0]
            curr_y = c[1]

            center_str = f"{c[0]}_{c[1]}"
            if center_str not in counted_centers:
                
                #cambiar si la linea es vertical
                if (prev_y < custom_y1 and curr_y >= custom_y1) and (curr_x > custom_x1 and curr_x < custom_x2):
                    count += 1
                    counted_centers.add(center_str)
                elif (prev_y > custom_y1 and curr_y <= custom_y1) and (curr_x > custom_x1 and curr_x < custom_x2):
                    count -= 1
                    counted_centers.add(center_str)

    previous_centers = current_centers.copy()

    cv2.putText(annotated_frame, f"Ocupados: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


    cv2.imshow("Solo Autos", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
