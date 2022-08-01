import cv2
import numpy as np

net = cv2.dnn.readNet("./yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))


cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
while True:
    _, frame = cap.read()

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detected_obj = net.forward()