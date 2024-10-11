import cv2


cap = cv2.VideoCapture("../surveil_137.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, 175000)
from ultralytics import YOLOv10

ov_qmodel = YOLOv10("/home/ubuntu/yolov10/int8/yolov10x_openvino_model/")
while True:
    # Leer el siguiente frame
    print("Funcionando")
    ret, frame = cap.read()
    if not ret:
        print("No se pudo obtener el frame. Fin del video o error.")
        break
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
