import cv2
import numpy as np
from ultralytics import YOLO
import math
import json
import time

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Reducir el tamaño del frame (ajústalo si quieres más velocidad)
resize_factor = 0.3

# Cargar coordenadas
with open("coordinates.json", 'r') as file:
    coordinates = json.load(file)

list_of_coordinates = [(int(coord['x']), int(coord['y'])) for coord in coordinates]

c_x1, c_y1 = list_of_coordinates[0]
c_x2, c_y2 = list_of_coordinates[1]

custom_x1, custom_y1 = int(c_x1 * resize_factor), int(c_y1 * resize_factor)
custom_x2, custom_y2 = int(c_x2 * resize_factor), int(c_y2 * resize_factor)

# Cargar modelo YOLO optimizado
model = YOLO("yolov8n.pt")  # o "yolov8n.engine" si usas TensorRT
model.fuse()  # Acelera la inferencia

# Abrir video
video_path = "/Users/caballero/Desktop/Servicio Social/carCounter/video.mp4"
cap = cv2.VideoCapture(video_path)

# Contadores
count = 0
previous_centers = []
counted_centers = set()
distance_threshold = 50  # píxeles

while cap.isOpened():
    start_time = time.time()

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    annotated_frame = frame.copy()

    results = model.predict(frame, stream=False, classes=[2], verbose=False)  # clase 2 = carro

    current_centers = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            score = boxes.conf[i].item()
            label = f"Car {score:.2f}"

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_centers.append((cx, cy))

            # Dibujar caja y centro
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(annotated_frame, (cx, cy), 3, (255, 0, 0), -1)

    # Dibujar línea
    cv2.line(annotated_frame, (custom_x1, custom_y1), (custom_x2, custom_y2), (0, 0, 255), 2)

    for c in current_centers:
        min_dist = float('inf')
        closest_prev = None
        for pc in previous_centers:
            dist = euclidean_distance(c, pc)
            if dist < min_dist:
                min_dist = dist
                closest_prev = pc

        if min_dist < distance_threshold and closest_prev is not None:
            prev_x, prev_y = closest_prev
            curr_x, curr_y = c

            center_str = f"{c[0]}_{c[1]}"
            if center_str not in counted_centers:
                if (prev_y < custom_y1 and curr_y >= custom_y1) and (custom_x1 < curr_x < custom_x2):
                    count += 1
                    counted_centers.add(center_str)
                elif (prev_y > custom_y1 and curr_y <= custom_y1) and (custom_x1 < curr_x < custom_x2):
                    count -= 1
                    counted_centers.add(center_str)

    previous_centers = current_centers.copy()

    cv2.putText(annotated_frame, f"Ocupados: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Mostrar FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("Solo Autos", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
