import cv2 as cv
from ultralytics import YOLO

camera = cv.VideoCapture(0)
model = YOLO("yolo26m_original.engine")
device = "cuda:0"

while True:
    ret, frame = camera.read()
    if not ret:
        break

    results = model(frame, device=device, verbose=False)

    for object in results[0].boxes:
        x1, y1, x2, y2 = map(int, object.xyxy[0])
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, f"Class: {int(object.cls[0])}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imshow('Camera Feed', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv.destroyAllWindows()