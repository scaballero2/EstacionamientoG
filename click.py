import cv2
import json


coordenadas = []

video_path = "/Users/caballero/Desktop/Servicio Social/carCounter/video.mp4"

cap = cv2.VideoCapture(video_path)

def click_event(event, x, y, flags, param):
    global counter, coordenadas
    if coordenadas.__len__() >= 2:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        data = {
            f'x': x,
            f'y': y
        }
        coordenadas.append(data)
        print(f"Coordenadas: ({x}, {y})")

        
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', click_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

with open('coordinates.json', 'w') as f:
    json.dump(coordenadas, f)
